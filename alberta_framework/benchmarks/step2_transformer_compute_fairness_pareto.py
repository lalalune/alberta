#!/usr/bin/env python3
"""Compute-fairness Pareto smoke report for Step 2 transformer memory.

This script does not train models. It reads existing result artifacts from the
advantage-memory transformer runner and throughput benchmark, then writes a
small set of tables for compute-fairness planning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

DEFAULT_TOKEN_ARTIFACTS = (
    Path(
        "outputs/step2_new_directions/"
        "advantage_memory_transformer_confirmatory_validation_30seed/"
        "validation_3000_30seed_eval4096_fw512_eb512_replay128_scalar_glr05_l2_01_gmax015/"
        "results.json"
    ),
    Path(
        "outputs/step2_new_directions/"
        "advantage_memory_transformer_3000_10seed_replay128_scalar_glr05_l2_01_gmax015/"
        "results.json"
    ),
    Path(
        "outputs/step2_new_directions/"
        "advantage_memory_transformer_5000_10seed_replay128_scalar_glr05_l2_01_gmax015/"
        "results.json"
    ),
    Path(
        "outputs/step2_new_directions/"
        "advantage_memory_transformer_10000_10seed_replay128_scalar_glr05_l2_01_gmax015/"
        "results.json"
    ),
)

DEFAULT_THROUGHPUT_ARTIFACTS = (
    Path("outputs/step2_new_directions/replay_cache_ablation_p64/results.json"),
    Path("outputs/step2_new_directions/replay_cache_ablation_p256/results.json"),
)

DEFAULT_STRONGER_FFN_ARTIFACT = Path(
    "outputs/step2_new_directions/stronger_ffn_validation_10000_30seed_best2/results.json"
)

TOKEN_BASELINE = "baseline_ffn_transformer"
THROUGHPUT_BASELINE = "baseline_ffn"
PRIMARY_CANDIDATE = "advantage_post_ffn_memory"
DEFAULT_EFFECT_THRESHOLD_NLL = 0.005

METHOD_LABELS = {
    "baseline_ffn_transformer": "Baseline FFN",
    "advantage_post_ffn_memory": "Exact replay memory, post-FFN",
    "advantage_pre_ffn_kv_memory": "Exact replay memory, pre-FFN KV",
    "baseline_ffn": "Baseline FFN",
    "advantage_post_current_exact_reference": "Current-token memory",
    "advantage_post_replay_exact_reference": "Exact replay reference",
    "advantage_post_replay_exact_fused_center": "Exact replay fused center",
    "advantage_post_replay_cached_basis_ablation": "Cached replay ablation",
    "advantage_pre_kv_replay_exact_reference": "Exact replay pre-FFN KV",
}

LOWER_IS_BETTER = {
    "final_window_nll",
    "eval_nll",
    "eval_perplexity",
    "train_s",
    "steady_mean_s",
    "compile_plus_first_s",
}


@dataclass(frozen=True)
class MetricStats:
    """Mean and uncertainty for one metric."""

    n: int
    mean: float
    stderr: float


@dataclass(frozen=True)
class TokenBudgetRow:
    """Quality row at equal online token budget."""

    source: str
    steps: int
    seeds: int
    eval_steps: int
    method: str
    label: str
    trainable_params: int
    state_bytes: int
    final_window_nll: float
    final_window_nll_stderr: float
    eval_nll: float
    eval_nll_stderr: float
    final_window_diff_vs_baseline: float | None
    final_window_diff_stderr: float | None
    eval_diff_vs_baseline: float | None
    eval_diff_stderr: float | None
    effect_threshold_nll: float
    eval_threshold_margin: float | None


@dataclass(frozen=True)
class ResourceGapRow:
    """Current resource mismatch against the FFN baseline."""

    source: str
    steps: int
    method: str
    label: str
    baseline_params: int
    method_params: int
    params_delta: int
    baseline_state_bytes: int
    method_state_bytes: int
    state_bytes_delta: int
    equal_params_status: str
    equal_state_status: str


@dataclass(frozen=True)
class HotLoopRow:
    """Throughput and quality row with compile separated from hot-loop time."""

    source: str
    proto_count: int
    steps: int
    method: str
    label: str
    behavior_contract: str
    center_update_path: str
    replay_feature_source: str
    trainable_params: int
    state_bytes: int
    compile_plus_first_s: float
    steady_mean_s: float
    steady_stderr_s: float
    steady_steps_per_s: float
    hot_slowdown_vs_ffn: float
    baseline_tokens_at_equal_hot_time: float
    same_token_final_diff_vs_ffn: float | None
    same_token_eval_diff_vs_ffn: float | None
    eval_threshold_margin: float | None
    final_window_nll: float
    eval_nll: float
    eval_perplexity: float
    forward_macs_per_step: int
    prototype_distance_passes_per_step: int


@dataclass(frozen=True)
class ParetoRow:
    """Dominance summary for one throughput row."""

    source: str
    method: str
    label: str
    behavior_contract: str
    nondominated: bool
    dominated_by: str
    note: str


@dataclass(frozen=True)
class StrongerFfnRow:
    """Validation-tuned FFN row compared against transformer memory."""

    source: str
    method: str
    n: int
    final_window_nll: float
    final_window_nll_stderr: float
    eval_nll: float
    eval_nll_stderr: float
    eval_perplexity: float
    eval_perplexity_stderr: float
    train_steps_per_s: float
    post_memory_eval_advantage: float | None
    post_memory_final_advantage: float | None
    pre_memory_eval_advantage: float | None
    pre_memory_final_advantage: float | None


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, Any], parsed)


def _float(value: Any) -> float:
    return float(cast(int | float, value))


def _int(value: Any) -> int:
    return int(cast(int | float, value))


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _mean_stderr(values: Iterable[float]) -> MetricStats:
    data = list(values)
    if not data:
        raise ValueError("cannot summarize an empty sequence")
    mean = sum(data) / len(data)
    if len(data) == 1:
        return MetricStats(n=1, mean=mean, stderr=0.0)
    variance = sum((value - mean) ** 2 for value in data) / (len(data) - 1)
    return MetricStats(n=len(data), mean=mean, stderr=math.sqrt(variance / len(data)))


def _records_by_method(payload: dict[str, Any]) -> dict[str, dict[int, dict[str, float]]]:
    grouped: dict[str, dict[int, dict[str, float]]] = {}
    records = cast(list[dict[str, Any]], payload["records"])
    for record in records:
        method = str(record["method"])
        seed = _int(record["seed"])
        summary = cast(dict[str, Any], record["summary"])
        grouped.setdefault(method, {})[seed] = {
            str(key): _float(value) for key, value in summary.items()
        }
    return grouped


def _paired_diff(
    grouped: dict[str, dict[int, dict[str, float]]],
    method: str,
    metric: str,
    *,
    baseline: str,
) -> MetricStats:
    baseline_records = grouped[baseline]
    method_records = grouped[method]
    seeds = sorted(set(baseline_records).intersection(method_records))
    if not seeds:
        raise ValueError(f"no paired seeds for {method} vs {baseline}")
    diffs = []
    for seed in seeds:
        base_value = baseline_records[seed][metric]
        method_value = method_records[seed][metric]
        if metric in LOWER_IS_BETTER:
            diffs.append(base_value - method_value)
        else:
            diffs.append(method_value - base_value)
    return _mean_stderr(diffs)


def _profile(payload: dict[str, Any], method: str) -> dict[str, Any]:
    profiles = cast(dict[str, dict[str, Any]], payload["profiles"])
    return profiles[method]


def token_budget_rows(
    paths: Iterable[Path],
    *,
    effect_threshold_nll: float,
) -> tuple[list[TokenBudgetRow], list[ResourceGapRow]]:
    """Build equal-token quality rows and resource-gap rows."""
    quality_rows: list[TokenBudgetRow] = []
    resource_rows: list[ResourceGapRow] = []
    for path in paths:
        payload = _load_json(path)
        config = cast(dict[str, Any], payload["config"])
        grouped = _records_by_method(payload)
        source = path.parent.name
        steps = _int(config["steps"])
        eval_steps = _int(config["eval_steps"])
        baseline_profile = _profile(payload, TOKEN_BASELINE)
        baseline_params = _int(baseline_profile["trainable_params"])
        baseline_state_bytes = _int(baseline_profile["state_bytes"])
        for method in sorted(grouped):
            profile = _profile(payload, method)
            final_stats = _mean_stderr(
                record["final_window_nll"] for record in grouped[method].values()
            )
            eval_stats = _mean_stderr(record["eval_nll"] for record in grouped[method].values())
            final_diff: MetricStats | None = None
            eval_diff: MetricStats | None = None
            if method != TOKEN_BASELINE:
                final_diff = _paired_diff(
                    grouped,
                    method,
                    "final_window_nll",
                    baseline=TOKEN_BASELINE,
                )
                eval_diff = _paired_diff(
                    grouped,
                    method,
                    "eval_nll",
                    baseline=TOKEN_BASELINE,
                )
            quality_rows.append(
                TokenBudgetRow(
                    source=source,
                    steps=steps,
                    seeds=final_stats.n,
                    eval_steps=eval_steps,
                    method=method,
                    label=_label(method),
                    trainable_params=_int(profile["trainable_params"]),
                    state_bytes=_int(profile["state_bytes"]),
                    final_window_nll=final_stats.mean,
                    final_window_nll_stderr=final_stats.stderr,
                    eval_nll=eval_stats.mean,
                    eval_nll_stderr=eval_stats.stderr,
                    final_window_diff_vs_baseline=None
                    if final_diff is None
                    else final_diff.mean,
                    final_window_diff_stderr=None
                    if final_diff is None
                    else final_diff.stderr,
                    eval_diff_vs_baseline=None if eval_diff is None else eval_diff.mean,
                    eval_diff_stderr=None if eval_diff is None else eval_diff.stderr,
                    effect_threshold_nll=effect_threshold_nll,
                    eval_threshold_margin=None
                    if eval_diff is None
                    else eval_diff.mean - effect_threshold_nll,
                )
            )
        for method in sorted(grouped):
            if method == TOKEN_BASELINE:
                continue
            profile = _profile(payload, method)
            method_params = _int(profile["trainable_params"])
            method_state_bytes = _int(profile["state_bytes"])
            resource_rows.append(
                ResourceGapRow(
                    source=source,
                    steps=steps,
                    method=method,
                    label=_label(method),
                    baseline_params=baseline_params,
                    method_params=method_params,
                    params_delta=method_params - baseline_params,
                    baseline_state_bytes=baseline_state_bytes,
                    method_state_bytes=method_state_bytes,
                    state_bytes_delta=method_state_bytes - baseline_state_bytes,
                    equal_params_status="missing parameter-matched FFN artifact",
                    equal_state_status="missing state-matched baseline artifact",
                )
            )
    return quality_rows, resource_rows


def _throughput_summary(row: dict[str, Any], metric: str) -> float:
    summary = cast(dict[str, Any], row["summary"])
    return _float(summary[metric])


def hot_loop_rows(
    paths: Iterable[Path],
    *,
    effect_threshold_nll: float,
) -> list[HotLoopRow]:
    """Build hot-loop rows from throughput benchmark artifacts."""
    rows: list[HotLoopRow] = []
    for path in paths:
        payload = _load_json(path)
        config = cast(dict[str, Any], payload["config"])
        results = cast(list[dict[str, Any]], payload["results"])
        baseline = next(row for row in results if row["name"] == THROUGHPUT_BASELINE)
        baseline_sps = _float(baseline["steady_steps_per_s"])
        source = path.parent.name
        for row in results:
            name = str(row["name"])
            same_token_final_diff: float | None = None
            same_token_eval_diff: float | None = None
            threshold_margin: float | None = None
            if name != THROUGHPUT_BASELINE:
                same_token_final_diff = _throughput_summary(
                    baseline,
                    "final_window_nll",
                ) - _throughput_summary(row, "final_window_nll")
                same_token_eval_diff = _throughput_summary(
                    baseline,
                    "eval_nll",
                ) - _throughput_summary(row, "eval_nll")
                threshold_margin = same_token_eval_diff - effect_threshold_nll
            steady_steps_per_s = _float(row["steady_steps_per_s"])
            rows.append(
                HotLoopRow(
                    source=source,
                    proto_count=_int(config["proto_count"]),
                    steps=_int(config["steps"]),
                    method=name,
                    label=_label(name),
                    behavior_contract=str(row["behavior_contract"]),
                    center_update_path=str(row["center_update_path"]),
                    replay_feature_source=str(row["replay_feature_source"]),
                    trainable_params=_int(row["trainable_params"]),
                    state_bytes=_int(row["state_bytes"]),
                    compile_plus_first_s=_float(row["compile_plus_first_s"]),
                    steady_mean_s=_float(row["steady_mean_s"]),
                    steady_stderr_s=_float(row["steady_stderr_s"]),
                    steady_steps_per_s=steady_steps_per_s,
                    hot_slowdown_vs_ffn=baseline_sps / steady_steps_per_s,
                    baseline_tokens_at_equal_hot_time=baseline_sps * _float(row["steady_mean_s"]),
                    same_token_final_diff_vs_ffn=same_token_final_diff,
                    same_token_eval_diff_vs_ffn=same_token_eval_diff,
                    eval_threshold_margin=threshold_margin,
                    final_window_nll=_throughput_summary(row, "final_window_nll"),
                    eval_nll=_throughput_summary(row, "eval_nll"),
                    eval_perplexity=_throughput_summary(row, "eval_perplexity"),
                    forward_macs_per_step=_int(row["forward_macs_per_step"]),
                    prototype_distance_passes_per_step=_int(
                        row["prototype_distance_passes_per_step"],
                    ),
                )
            )
    return rows


def _dominates(left: HotLoopRow, right: HotLoopRow) -> bool:
    left_values = (
        left.final_window_nll,
        left.eval_nll,
        left.trainable_params,
        left.state_bytes,
        left.steady_mean_s,
    )
    right_values = (
        right.final_window_nll,
        right.eval_nll,
        right.trainable_params,
        right.state_bytes,
        right.steady_mean_s,
    )
    return all(a <= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a < b for a, b in zip(left_values, right_values, strict=True)
    )


def pareto_rows(rows: Iterable[HotLoopRow]) -> list[ParetoRow]:
    """Compute simple dominance over smoke quality, params, state, and hot time."""
    all_rows = list(rows)
    result: list[ParetoRow] = []
    for row in all_rows:
        same_source = [candidate for candidate in all_rows if candidate.source == row.source]
        dominators = [candidate for candidate in same_source if _dominates(candidate, row)]
        if dominators:
            first = dominators[0]
            result.append(
                ParetoRow(
                    source=row.source,
                    method=row.method,
                    label=row.label,
                    behavior_contract=row.behavior_contract,
                    nondominated=False,
                    dominated_by=first.method,
                    note=(
                        "Dominated on smoke final NLL, eval NLL, params, state bytes, "
                        "and hot-loop time."
                    ),
                )
            )
            continue
        note = "Nondominated in this smoke table."
        if "cached" in row.behavior_contract:
            note = (
                "Nondominated/dominated status is not production evidence; cached replay "
                "changes behavior."
            )
        result.append(
            ParetoRow(
                source=row.source,
                method=row.method,
                label=row.label,
                behavior_contract=row.behavior_contract,
                nondominated=True,
                dominated_by="",
                note=note,
            )
        )
    return result


def stronger_ffn_rows(path: Path | None) -> list[StrongerFfnRow]:
    """Read stronger validation-tuned FFN artifacts if present."""
    if path is None or not path.exists():
        return []
    payload = _load_json(path)
    aggregate_rows = cast(list[dict[str, Any]], payload.get("aggregate", []))
    memory_means = cast(dict[str, dict[str, float]], payload.get("memory_reference_means", {}))
    post = memory_means.get("advantage_post_ffn_memory", {})
    pre = memory_means.get("advantage_pre_ffn_kv_memory", {})
    rows: list[StrongerFfnRow] = []
    for row in aggregate_rows:
        final_nll = _float(row["final_window_nll_mean"])
        eval_nll = _float(row["eval_nll_mean"])
        rows.append(
            StrongerFfnRow(
                source=path.parent.name,
                method=str(row["method"]),
                n=_int(row["n"]),
                final_window_nll=final_nll,
                final_window_nll_stderr=_float(row["final_window_nll_stderr"]),
                eval_nll=eval_nll,
                eval_nll_stderr=_float(row["eval_nll_stderr"]),
                eval_perplexity=_float(row["eval_perplexity_mean"]),
                eval_perplexity_stderr=_float(row["eval_perplexity_stderr"]),
                train_steps_per_s=_float(row["train_steps_per_s_mean"]),
                post_memory_eval_advantage=None
                if "eval_nll" not in post
                else _float(post["eval_nll"]) - eval_nll,
                post_memory_final_advantage=None
                if "final_window_nll" not in post
                else _float(post["final_window_nll"]) - final_nll,
                pre_memory_eval_advantage=None
                if "eval_nll" not in pre
                else _float(pre["eval_nll"]) - eval_nll,
                pre_memory_final_advantage=None
                if "final_window_nll" not in pre
                else _float(pre["final_window_nll"]) - final_nll,
            )
        )
    return rows


def _write_csv(path: Path, rows: Iterable[Any]) -> None:
    row_list = [asdict(row) for row in rows]
    if not row_list:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_list[0]))
        writer.writeheader()
        writer.writerows(row_list)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _find_row(rows: Iterable[HotLoopRow], source: str, method: str) -> HotLoopRow | None:
    for row in rows:
        if row.source == source and row.method == method:
            return row
    return None


def _best_primary_token_rows(rows: Iterable[TokenBudgetRow]) -> list[TokenBudgetRow]:
    return [
        row
        for row in rows
        if row.method == PRIMARY_CANDIDATE
        and (
            row.source.startswith("validation_3000")
            or row.source.startswith("advantage_memory_transformer_")
        )
    ]


def write_summary(
    path: Path,
    *,
    token_rows: list[TokenBudgetRow],
    resource_rows: list[ResourceGapRow],
    hot_rows: list[HotLoopRow],
    dominance_rows: list[ParetoRow],
    stronger_rows: list[StrongerFfnRow],
    effect_threshold_nll: float,
) -> None:
    """Write the Markdown smoke summary."""
    lines = [
        "# Step 2 Transformer Compute-Fairness Pareto Smoke",
        "",
        "This smoke report reads existing artifacts only. It does not train models.",
        "Positive NLL diffs favor the memory method. Steady hot-loop timings exclude "
        "compile+first-run time.",
        "",
        f"Effect threshold for practical held-out NLL improvement: `{effect_threshold_nll}`.",
        "",
        "## Equal Token Budget",
        "",
        "| Source | Steps | Seeds | Eval steps | Method | Params | State bytes | "
        "Final NLL | Eval NLL | Eval diff vs FFN | Threshold margin |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for token_row in _best_primary_token_rows(token_rows):
        lines.append(
            f"| `{token_row.source}` | {token_row.steps} | {token_row.seeds} | "
            f"{token_row.eval_steps} | {token_row.label} | "
            f"{token_row.trainable_params} | {token_row.state_bytes} | "
            f"{token_row.final_window_nll:.4f} +/- "
            f"{token_row.final_window_nll_stderr:.4f} | "
            f"{token_row.eval_nll:.4f} +/- {token_row.eval_nll_stderr:.4f} | "
            f"{_fmt(token_row.eval_diff_vs_baseline, 6)} +/- "
            f"{_fmt(token_row.eval_diff_stderr, 6)} | "
            f"{_fmt(token_row.eval_threshold_margin, 6)} |"
        )
    lines.extend(
        [
            "",
            "## Equal Params And State Bytes",
            "",
            "| Source | Steps | Method | Params delta vs FFN | State bytes delta vs FFN | Status |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    seen_resource_keys: set[tuple[int, str]] = set()
    for resource_row in resource_rows:
        key = (resource_row.steps, resource_row.method)
        if key in seen_resource_keys:
            continue
        seen_resource_keys.add(key)
        lines.append(
            f"| `{resource_row.source}` | {resource_row.steps} | {resource_row.label} | "
            f"{resource_row.params_delta:+d} | {resource_row.state_bytes_delta:+d} | "
            f"{resource_row.equal_params_status}; {resource_row.equal_state_status} |"
        )
    lines.extend(
        [
            "",
            "## Equal Hot-Loop Wall Clock",
            "",
            "| Source | P | Method | Contract | Compile+first s | Hot s | Steps/s | "
            "Slowdown vs FFN | FFN tokens in same hot time | Same-token eval diff |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for hot_row in hot_rows:
        if hot_row.method not in {
            THROUGHPUT_BASELINE,
            "advantage_post_replay_exact_reference",
            "advantage_post_replay_exact_fused_center",
            "advantage_post_replay_cached_basis_ablation",
        }:
            continue
        lines.append(
            f"| `{hot_row.source}` | {hot_row.proto_count} | {hot_row.label} | "
            f"`{hot_row.behavior_contract}` | {hot_row.compile_plus_first_s:.4f} | "
            f"{hot_row.steady_mean_s:.4f} +/- {hot_row.steady_stderr_s:.4f} | "
            f"{hot_row.steady_steps_per_s:.1f} | {hot_row.hot_slowdown_vs_ffn:.2f}x | "
            f"{hot_row.baseline_tokens_at_equal_hot_time:.1f} | "
            f"{_fmt(hot_row.same_token_eval_diff_vs_ffn, 6)} |"
        )
    lines.extend(
        [
            "",
            "The equal-hot table is a smoke estimate, not a quality-at-equal-wall-clock "
            "claim. It shows how many FFN tokens fit in the memory method's hot time; "
            "the existing artifacts do not contain quality curves at those larger FFN "
            "token counts.",
            "",
            "## Quality To Threshold",
            "",
            "The 30-seed validation 3000-step artifact and the 10-seed 3000/5000-step "
            "artifacts do not clear the practical held-out NLL threshold of "
            f"`{effect_threshold_nll}` for the primary post-FFN memory candidate. "
            "The 10-seed 10000-step exploratory artifact clears the mean threshold, "
            "but it has only 512 held-out eval contexts and large paired uncertainty, "
            "so it is not a production threshold-crossing result.",
            "",
            "The artifacts also do not contain per-step quality curves, so they cannot "
            "answer first-token or first-hot-second-to-threshold. A production run must "
            "store rolling prequential and held-out checkpoints.",
            "",
            "## Pareto Smoke",
            "",
            "| Source | Method | Contract | Nondominated | Dominated by | Note |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for dominance_row in dominance_rows:
        lines.append(
            f"| `{dominance_row.source}` | {dominance_row.label} | "
            f"`{dominance_row.behavior_contract}` | {dominance_row.nondominated} | "
            f"`{dominance_row.dominated_by}` | {dominance_row.note} |"
        )
    p64_fused = _find_row(
        hot_rows,
        "replay_cache_ablation_p64",
        "advantage_post_replay_exact_fused_center",
    )
    p64_cached = _find_row(
        hot_rows,
        "replay_cache_ablation_p64",
        "advantage_post_replay_cached_basis_ablation",
    )
    if p64_fused is not None and p64_cached is not None:
        lines.extend(
            [
                "",
                "## Fused Center Versus Cached Replay",
                "",
                "Fused center is the production-safe path because the throughput benchmark's "
                "exactness check shows it preserves the current raw-context replay output. "
                "Cached replay is faster in the P=64 smoke, but it stores stale `basis_input` "
                "and base loss, changes the gate signal, and has worse same-token eval NLL "
                f"than exact fused center by {p64_cached.eval_nll - p64_fused.eval_nll:.4f}.",
            ]
        )
    if stronger_rows:
        lines.extend(
            [
                "",
                "## Stronger FFN Validation Baselines",
                "",
                "Positive memory-advantage columns mean the FFN row has lower NLL than "
                "the corresponding replay-memory candidate.",
                "",
                "| Source | Method | N | Final NLL | Eval NLL | Eval PPL | "
                "Post-memory eval advantage | Post-memory final advantage | Steps/s |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in stronger_rows:
            lines.append(
                f"| `{row.source}` | `{row.method}` | {row.n} | "
                f"{row.final_window_nll:.4f} +/- {row.final_window_nll_stderr:.4f} | "
                f"{row.eval_nll:.4f} +/- {row.eval_nll_stderr:.4f} | "
                f"{row.eval_perplexity:.2f} +/- {row.eval_perplexity_stderr:.2f} | "
                f"{_fmt(row.post_memory_eval_advantage, 6)} | "
                f"{_fmt(row.post_memory_final_advantage, 6)} | "
                f"{row.train_steps_per_s:.1f} |"
            )
        lines.extend(
            [
                "",
                "These rows overturn the narrow exploratory memory claim. The tuned "
                "FFN h=96/h=128 validation baselines are materially better on held-out "
                "NLL than the replay-memory candidates, and h=96/h=128 with lr=0.1 "
                "also matches or slightly beats the post-FFN memory final-window NLL. "
                "No lockbox run is justified until a new candidate beats these rows.",
            ]
        )
    lines.extend(
        [
            "",
            "## Production Benchmark Remaining",
            "",
            "Run a frozen exact-replay benchmark with fused center update, 30 paired seeds, "
            "train/validation/lockbox byte ranges, parameter-matched FFN, state-matched "
            "control, equal-hot-loop stopping, equal-token stopping, and rolling quality "
            "curves for threshold crossing. Cached replay should remain a named ablation, "
            "not the production replay result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-artifact",
        action="append",
        type=Path,
        dest="token_artifacts",
        help="Advantage-memory transformer results.json artifact.",
    )
    parser.add_argument(
        "--throughput-artifact",
        action="append",
        type=Path,
        dest="throughput_artifacts",
        help="Transformer memory throughput results.json artifact.",
    )
    parser.add_argument(
        "--stronger-ffn-artifact",
        type=Path,
        default=DEFAULT_STRONGER_FFN_ARTIFACT,
        help="Validation-tuned stronger FFN results.json artifact.",
    )
    parser.add_argument(
        "--effect-threshold-nll",
        type=float,
        default=DEFAULT_EFFECT_THRESHOLD_NLL,
        help="Smallest practical held-out NLL improvement.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_new_directions/compute_fairness_pareto_smoke"),
    )
    return parser.parse_args()


def main() -> None:
    """Generate the compute-fairness smoke report."""
    args = parse_args()
    token_paths = tuple(args.token_artifacts or DEFAULT_TOKEN_ARTIFACTS)
    throughput_paths = tuple(args.throughput_artifacts or DEFAULT_THROUGHPUT_ARTIFACTS)
    stronger_path = args.stronger_ffn_artifact
    required_paths = [*token_paths, *throughput_paths]
    if stronger_path is not None:
        required_paths.append(stronger_path)
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing input artifacts:\n{missing_text}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    token_rows, resource_rows = token_budget_rows(
        token_paths,
        effect_threshold_nll=args.effect_threshold_nll,
    )
    hot_rows = hot_loop_rows(
        throughput_paths,
        effect_threshold_nll=args.effect_threshold_nll,
    )
    dominance_rows = pareto_rows(hot_rows)
    stronger_rows = stronger_ffn_rows(stronger_path)

    _write_csv(args.output_dir / "token_budget_quality.csv", token_rows)
    _write_csv(args.output_dir / "resource_fairness_gaps.csv", resource_rows)
    _write_csv(args.output_dir / "hot_loop_quality.csv", hot_rows)
    _write_csv(args.output_dir / "pareto_smoke.csv", dominance_rows)
    _write_csv(args.output_dir / "stronger_ffn_quality.csv", stronger_rows)

    payload = {
        "effect_threshold_nll": args.effect_threshold_nll,
        "token_artifacts": [str(path) for path in token_paths],
        "throughput_artifacts": [str(path) for path in throughput_paths],
        "stronger_ffn_artifact": None if stronger_path is None else str(stronger_path),
        "token_budget_quality": [asdict(row) for row in token_rows],
        "resource_fairness_gaps": [asdict(row) for row in resource_rows],
        "hot_loop_quality": [asdict(row) for row in hot_rows],
        "pareto_smoke": [asdict(row) for row in dominance_rows],
        "stronger_ffn_quality": [asdict(row) for row in stronger_rows],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_summary(
        args.output_dir / "SUMMARY.md",
        token_rows=token_rows,
        resource_rows=resource_rows,
        hot_rows=hot_rows,
        dominance_rows=dominance_rows,
        stronger_rows=stronger_rows,
        effect_threshold_nll=args.effect_threshold_nll,
    )
    print(f"wrote {args.output_dir / 'SUMMARY.md'}")
    print(f"wrote {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()

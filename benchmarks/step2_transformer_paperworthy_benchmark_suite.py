#!/usr/bin/env python3
"""Paperworthy report generator for the Step 2 transformer memory benchmark.

This script does not run training. It reads existing
``step2_tiny_shakespeare_advantage_memory_transformer.py`` ``results.json``
artifacts and writes a reproducibility manifest, paired statistical tables,
and a protocol audit for the replay-capped advantage-memory transformer claim.

The default inputs are the current 10-seed replay-capped runs at 3000, 5000,
and 10000 online steps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats

DEFAULT_RESULT_PATHS = (
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

BASELINE_METHOD = "baseline_ffn_transformer"
DEFAULT_PRIMARY_METHOD = "advantage_post_ffn_memory"
DEFAULT_CANDIDATE_METHODS = (
    "advantage_post_ffn_memory",
    "advantage_pre_ffn_kv_memory",
)
DEFAULT_PRIMARY_METRICS = ("final_window_nll", "eval_nll")
DEFAULT_REPORT_METRICS = (
    "final_window_nll",
    "final_window_base_nll",
    "eval_nll",
    "eval_perplexity",
    "eval_fast_nll",
    "eval_fast_perplexity",
    "final_window_accuracy",
    "eval_accuracy",
    "train_s",
    "train_steps_per_s",
)

METHOD_LABELS = {
    "baseline_ffn_transformer": "Baseline FFN",
    "advantage_post_ffn_memory": "Replay-capped post-FFN memory",
    "advantage_pre_ffn_kv_memory": "Replay-capped pre-FFN KV memory",
}

LOWER_IS_BETTER = {
    "final_window_nll",
    "final_window_base_nll",
    "eval_nll",
    "eval_fast_nll",
    "eval_perplexity",
    "eval_fast_perplexity",
    "train_s",
}

METRIC_FALLBACKS = {
    "final_window_base_nll": "final_window_nll",
    "eval_fast_nll": "eval_nll",
    "eval_fast_perplexity": "eval_perplexity",
}


@dataclass(frozen=True)
class PairedComparison:
    """Serializable paired-comparison row."""

    source: str
    horizon_steps: int
    method: str
    metric: str
    n_pairs: int
    baseline_mean: float
    candidate_mean: float
    diff_mean_positive_favors_candidate: float
    diff_stderr: float
    diff_ci95_low: float
    diff_ci95_high: float
    relative_diff_percent: float
    paired_cohens_dz: float
    paired_t_statistic: float
    paired_t_p: float
    wilcoxon_p: float
    sign_test_p: float
    wins: int
    losses: int
    ties: int
    holm_p_all_reported: float = 1.0
    holm_significant_all_reported: bool = False
    holm_p_primary_family: float | None = None
    holm_significant_primary_family: bool | None = None


@dataclass(frozen=True)
class AuditCheck:
    """One benchmark-rigor audit check."""

    area: str
    status: str
    check: str
    evidence: str
    action: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, Any], parsed)


def _run_git(args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def _format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def _records_by_method(payload: dict[str, Any]) -> dict[str, dict[int, dict[str, float]]]:
    grouped: dict[str, dict[int, dict[str, float]]] = {}
    for record in payload["records"]:
        method = str(record["method"])
        seed = int(record["seed"])
        summary = {str(k): float(v) for k, v in record["summary"].items()}
        grouped.setdefault(method, {})[seed] = summary
    return grouped


def _summary_value(summary: dict[str, float], metric: str) -> float:
    if metric in summary:
        return summary[metric]
    fallback = METRIC_FALLBACKS.get(metric)
    if fallback is not None and fallback in summary:
        return summary[fallback]
    raise KeyError(f"metric {metric!r} is absent and has no available fallback")


def _paired_arrays(
    payload: dict[str, Any],
    method: str,
    metric: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    grouped = _records_by_method(payload)
    baseline = grouped[BASELINE_METHOD]
    candidate = grouped[method]
    seeds = sorted(set(baseline).intersection(candidate))
    baseline_values = cast(
        NDArray[np.float64],
        np.asarray(
            [_summary_value(baseline[seed], metric) for seed in seeds],
            dtype=np.float64,
        ),
    )
    candidate_values = cast(
        NDArray[np.float64],
        np.asarray(
            [_summary_value(candidate[seed], metric) for seed in seeds],
            dtype=np.float64,
        ),
    )
    return baseline_values, candidate_values, seeds


def _diffs_positive_favor_candidate(
    baseline: NDArray[np.float64],
    candidate: NDArray[np.float64],
    metric: str,
) -> NDArray[np.float64]:
    if metric in LOWER_IS_BETTER:
        return baseline - candidate
    return candidate - baseline


def _stderr(values: NDArray[np.float64]) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _ci95(values: NDArray[np.float64]) -> tuple[float, float]:
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean
    margin = float(stats.t.ppf(0.975, values.size - 1) * _stderr(values))
    return mean - margin, mean + margin


def _paired_dz(diffs: NDArray[np.float64]) -> float:
    if diffs.size < 2:
        return 0.0
    std = float(np.std(diffs, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(diffs) / std)


def _safe_ttest_1samp(diffs: NDArray[np.float64]) -> tuple[float, float]:
    if diffs.size < 2 or float(np.std(diffs, ddof=1)) == 0.0:
        return 0.0, 1.0
    result = stats.ttest_1samp(diffs, popmean=0.0)
    return float(result.statistic), float(result.pvalue)


def _safe_wilcoxon(diffs: NDArray[np.float64]) -> float:
    if diffs.size < 2 or np.allclose(diffs, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _safe_sign_test(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    return float(stats.binomtest(wins, trials, p=0.5, alternative="two-sided").pvalue)


def _holm_adjust(p_values: Sequence[float], alpha: float) -> tuple[list[float], list[bool]]:
    """Return Holm-adjusted p-values and rejections in original order."""
    n = len(p_values)
    if n == 0:
        return [], []

    order = sorted(range(n), key=lambda idx: p_values[idx])
    adjusted_sorted = [0.0] * n
    previous = 0.0
    for rank, idx in enumerate(order):
        adjusted_p = min(1.0, (n - rank) * p_values[idx])
        previous = max(previous, adjusted_p)
        adjusted_sorted[rank] = previous

    adjusted: list[float] = [1.0] * n
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]

    rejected_sorted: list[bool] = []
    still_rejecting = True
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        reject = still_rejecting and p_values[idx] <= threshold
        rejected_sorted.append(reject)
        if not reject:
            still_rejecting = False

    rejected = [False] * n
    for flag, idx in zip(rejected_sorted, order, strict=True):
        rejected[idx] = flag
    return adjusted, rejected


def compare_payload(
    path: Path,
    payload: dict[str, Any],
    *,
    methods: Sequence[str],
    metrics: Sequence[str],
) -> list[PairedComparison]:
    comparisons: list[PairedComparison] = []
    steps = int(payload["config"]["steps"])
    for method in methods:
        for metric in metrics:
            baseline, candidate, _seeds = _paired_arrays(payload, method, metric)
            diffs = _diffs_positive_favor_candidate(baseline, candidate, metric)
            ci_low, ci_high = _ci95(diffs)
            t_stat, t_p = _safe_ttest_1samp(diffs)
            wins = int(np.sum(diffs > 1e-12))
            losses = int(np.sum(diffs < -1e-12))
            ties = int(diffs.size - wins - losses)
            baseline_mean = float(np.mean(baseline))
            diff_mean = float(np.mean(diffs))
            relative = 0.0
            if baseline_mean != 0.0:
                relative = 100.0 * diff_mean / abs(baseline_mean)
            comparisons.append(
                PairedComparison(
                    source=str(path),
                    horizon_steps=steps,
                    method=method,
                    metric=metric,
                    n_pairs=int(diffs.size),
                    baseline_mean=baseline_mean,
                    candidate_mean=float(np.mean(candidate)),
                    diff_mean_positive_favors_candidate=diff_mean,
                    diff_stderr=_stderr(diffs),
                    diff_ci95_low=ci_low,
                    diff_ci95_high=ci_high,
                    relative_diff_percent=relative,
                    paired_cohens_dz=_paired_dz(diffs),
                    paired_t_statistic=t_stat,
                    paired_t_p=t_p,
                    wilcoxon_p=_safe_wilcoxon(diffs),
                    sign_test_p=_safe_sign_test(wins, losses),
                    wins=wins,
                    losses=losses,
                    ties=ties,
                )
            )
    return comparisons


def apply_corrections(
    rows: Sequence[PairedComparison],
    *,
    primary_method: str,
    primary_metrics: Sequence[str],
    alpha: float,
) -> list[PairedComparison]:
    all_adjusted, all_rejected = _holm_adjust([row.paired_t_p for row in rows], alpha)
    primary_indices = [
        idx
        for idx, row in enumerate(rows)
        if row.method == primary_method and row.metric in primary_metrics
    ]
    primary_adjusted, primary_rejected = _holm_adjust(
        [rows[idx].paired_t_p for idx in primary_indices],
        alpha,
    )
    primary_by_index = {
        idx: (primary_adjusted[offset], primary_rejected[offset])
        for offset, idx in enumerate(primary_indices)
    }

    corrected: list[PairedComparison] = []
    for idx, row in enumerate(rows):
        primary = primary_by_index.get(idx)
        corrected.append(
            PairedComparison(
                **{
                    **asdict(row),
                    "holm_p_all_reported": all_adjusted[idx],
                    "holm_significant_all_reported": all_rejected[idx],
                    "holm_p_primary_family": None if primary is None else primary[0],
                    "holm_significant_primary_family": None
                    if primary is None
                    else primary[1],
                }
            )
        )
    return corrected


def _method_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in payload["records"]:
        counts[str(record["method"])] = counts.get(str(record["method"]), 0) + 1
    return counts


def _candidate_methods(payloads: Sequence[dict[str, Any]]) -> list[str]:
    methods = set(DEFAULT_CANDIDATE_METHODS)
    for payload in payloads:
        methods.update(_records_by_method(payload))
    methods.discard(BASELINE_METHOD)
    return [method for method in DEFAULT_CANDIDATE_METHODS if method in methods]


REQUIRED_OFFSET_KEYS = (
    "train_offset",
    "train_effective_offset",
    "eval_offset",
    "eval_effective_offset",
)


def _nested_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload.get("manifest", {})
    return manifest if isinstance(manifest, dict) else {}


def _payload_has_git(payload: dict[str, Any]) -> bool:
    return "git" in payload or "git" in _nested_manifest(payload)


def _payload_has_command(payload: dict[str, Any]) -> bool:
    manifest = _nested_manifest(payload)
    return (
        "command" in payload
        or "command" in manifest
        or bool(manifest.get("argv"))
    )


def _payload_has_environment(payload: dict[str, Any]) -> bool:
    return "environment" in payload or "environment" in _nested_manifest(payload)


def _payload_has_data_hash(payload: dict[str, Any]) -> bool:
    if "data_sha256" in payload or "data_hash" in payload:
        return True
    data = payload.get("data", {})
    if isinstance(data, dict) and (
        "sha256" in data or "data_sha256" in data or "data_hash" in data
    ):
        return True
    manifest_data = _nested_manifest(payload).get("data", {})
    return isinstance(manifest_data, dict) and (
        "sha256" in manifest_data
        or "data_sha256" in manifest_data
        or "data_hash" in manifest_data
    )


def _payload_has_hardware(payload: dict[str, Any]) -> bool:
    environment = _nested_manifest(payload).get("environment", {})
    if not isinstance(environment, dict):
        return False
    return isinstance(environment.get("platform"), dict) and isinstance(
        environment.get("jax"),
        dict,
    )


def _payload_has_jax_version(payload: dict[str, Any]) -> bool:
    environment = _nested_manifest(payload).get("environment", {})
    packages = environment.get("packages", {}) if isinstance(environment, dict) else {}
    return isinstance(packages, dict) and "jax" in packages


def _payload_has_nested_offsets(payload: dict[str, Any]) -> bool:
    manifest = _nested_manifest(payload)
    seed_runs_raw = manifest.get("seed_runs", [])
    seed_runs = {
        int(seed_run["seed_index"]): seed_run
        for seed_run in seed_runs_raw
        if isinstance(seed_run, dict) and "seed_index" in seed_run
    }
    if not seed_runs:
        return False
    for record in payload["records"]:
        data_offsets = record.get("data_offsets")
        if not isinstance(data_offsets, dict):
            return False
        if any(key not in data_offsets for key in REQUIRED_OFFSET_KEYS):
            return False
        seed_run = seed_runs.get(int(record["seed"]))
        methods = seed_run.get("methods", {}) if isinstance(seed_run, dict) else {}
        method_offsets = methods.get(str(record["method"])) if isinstance(methods, dict) else None
        if not isinstance(method_offsets, dict):
            return False
        if any(key not in method_offsets for key in REQUIRED_OFFSET_KEYS):
            return False
    return True


def audit_payloads(payloads: Sequence[dict[str, Any]]) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    min_seeds = min(int(payload["config"]["seeds"]) for payload in payloads)
    min_eval_steps = min(int(payload["config"]["eval_steps"]) for payload in payloads)
    baseline_counts = [
        sum(1 for method in _method_counts(payload) if method == BASELINE_METHOD)
        for payload in payloads
    ]
    has_single_baseline = all(count == 1 for count in baseline_counts)
    has_git = all(_payload_has_git(payload) for payload in payloads)
    has_command = all(_payload_has_command(payload) for payload in payloads)
    has_environment = all(_payload_has_environment(payload) for payload in payloads)
    has_data_hash = all(_payload_has_data_hash(payload) for payload in payloads)
    captures_offsets = all(_payload_has_nested_offsets(payload) for payload in payloads)
    configs = [payload["config"] for payload in payloads]
    config_keys = set().union(*(config.keys() for config in configs))
    missing_config_keys = {
        "proto_adaptive_bandwidth",
        "proto_bandwidth_update_rate",
    }.difference(config_keys)
    if not has_command:
        missing_config_keys.add("command")
    if not all(_payload_has_hardware(payload) for payload in payloads):
        missing_config_keys.add("hardware")
    if not all(_payload_has_jax_version(payload) for payload in payloads):
        missing_config_keys.add("jax_version")

    checks.extend(
        [
            AuditCheck(
                "data protocol",
                "FAIL",
                "Independent validation and lockbox test splits are required.",
                "Artifacts expose one train_fraction and one held-out eval slice.",
                "Freeze train/validation/test byte ranges before tuning; report lockbox once.",
            ),
            AuditCheck(
                "data protocol",
                "FAIL" if min_eval_steps < 2048 else "PARTIAL",
                "Held-out evaluation should be large enough for tiny effects.",
                f"Minimum eval_steps across inputs is {min_eval_steps}.",
                "Use thousands of held-out contexts per seed or a full fixed validation shard.",
            ),
            AuditCheck(
                "seeds",
                "FAIL" if min_seeds < 30 else "PASS",
                "Use enough paired seeds for sub-0.1% NLL effects.",
                f"Minimum seed count is {min_seeds}.",
                "Promote with at least 30 paired seeds or a power analysis.",
            ),
            AuditCheck(
                "paired statistics",
                "PASS",
                "Paired diffs, confidence intervals, tests, and effect sizes are generated.",
                "This report pairs methods by seed within each result artifact.",
                "Use this generated table instead of unpaired mean +/- stderr claims.",
            ),
            AuditCheck(
                "multiple comparisons",
                "PARTIAL",
                "Holm correction is applied to reported tables.",
                "The historical hyperparameter search is not part of the result artifacts.",
                "Separate exploratory sweeps from a preregistered confirmatory rerun.",
            ),
            AuditCheck(
                "config capture",
                "FAIL" if missing_config_keys else "PASS",
                "Artifacts should capture all flags, command, code revision, and environment.",
                "Missing payload fields: " + ", ".join(sorted(missing_config_keys)),
                "Write a run manifest from the training runner, not only from "
                "this post hoc report.",
            ),
            AuditCheck(
                "reproducibility",
                "FAIL" if (not has_git or not has_command or not has_environment) else "PASS",
                "Training artifacts should include git, command, and package versions.",
                f"git={has_git}, command={has_command}, environment={has_environment}.",
                "Embed git status, argv, JAX/JAXLIB versions, device, and Python dependencies.",
            ),
            AuditCheck(
                "reproducibility",
                "FAIL" if not has_data_hash else "PASS",
                "The corpus bytes used for training and evaluation need a hash.",
                f"Data hash present in payloads: {has_data_hash}.",
                "Store sha256 and byte count for Tiny Shakespeare in every run artifact.",
            ),
            AuditCheck(
                "repeatability",
                "FAIL" if not captures_offsets else "PASS",
                "Per-seed stream offsets and eval offsets should be captured.",
                f"Nested record and manifest offsets captured: {captures_offsets}.",
                "Record train/eval offsets in each record and per-seed manifest.",
            ),
            AuditCheck(
                "baselines",
                "FAIL" if has_single_baseline else "PARTIAL",
                "A single tuned FFN baseline is not enough for a paper claim.",
                "Current artifacts compare against one FFN transformer baseline.",
                "Add parameter-matched, compute-matched, replay-only, cap-only, "
                "and tuned no-memory baselines.",
            ),
            AuditCheck(
                "compute reporting",
                "PARTIAL",
                "Train seconds and steps/sec are recorded, but compile and hardware are not.",
                "Profiles include params/state bytes; timing lacks device and warmup split.",
                "Report hardware, backend, compile time, hot-loop time, peak memory, "
                "and energy if available.",
            ),
            AuditCheck(
                "failure modes",
                "PARTIAL",
                "Gate and advantage diagnostics are recorded for memory methods.",
                "No preregistered failure thresholds are present.",
                "Fail runs on NaNs, negative replay advantage with open gate, "
                "or held-out regression.",
            ),
        ]
    )
    return checks


def build_manifest(paths: Sequence[Path], payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    data_paths = sorted({str(payload["config"].get("data_path", "")) for payload in payloads})
    data_manifest = []
    for raw_path in data_paths:
        path = Path(raw_path)
        if path.exists():
            data_manifest.append(
                {
                    "path": raw_path,
                    "exists": True,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        else:
            data_manifest.append({"path": raw_path, "exists": False})

    source_manifest = []
    for path, payload in zip(paths, payloads, strict=True):
        source_manifest.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "config": payload["config"],
                "prototype_block": payload.get("prototype_block"),
                "profiles": payload.get("profiles"),
                "method_record_counts": _method_counts(payload),
            }
        )

    return {
        "schema": "alberta.step2.transformer_paperworthy_manifest.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator": Path(__file__).name,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": getattr(stats, "__version__", "unknown"),
        },
        "git": _git_metadata(),
        "data": data_manifest,
        "sources": source_manifest,
    }


def _command_from_config(config: dict[str, Any]) -> str:
    script = (
        '"examples/The Alberta Plan/Step2/'
        'step2_tiny_shakespeare_advantage_memory_transformer.py"'
    )
    flag_map = {
        "steps": "--steps",
        "seeds": "--seeds",
        "block_size": "--block-size",
        "d_model": "--d-model",
        "mlp_hidden": "--mlp-hidden",
        "proto_count": "--proto-count",
        "eval_steps": "--eval-steps",
        "final_window": "--final-window",
        "train_fraction": "--train-fraction",
        "baseline_lr": "--baseline-lr",
        "fast_lr": "--fast-lr",
        "slow_lr": "--slow-lr",
        "grad_clip": "--grad-clip",
        "proto_update_rate": "--proto-update-rate",
        "proto_novelty_threshold": "--proto-novelty-threshold",
        "proto_bandwidth": "--proto-bandwidth",
        "gate_init_logit": "--gate-init-logit",
        "gate_lr": "--gate-lr",
        "gate_decay": "--gate-decay",
        "gate_max": "--gate-max",
        "advantage_margin": "--advantage-margin",
        "gate_l2": "--gate-l2",
        "gate_mode": "--gate-mode",
        "gate_objective": "--gate-objective",
        "replay_size": "--replay-size",
        "train_loss_mode": "--train-loss-mode",
        "memory_loss_weight": "--memory-loss-weight",
        "reset_mode": "--reset-mode",
        "seed": "--seed",
        "data_path": "--data-path",
        "output_dir": "--output-dir",
    }
    args = ["source .venv/bin/activate && python", script]
    for key, flag in flag_map.items():
        if key in config:
            args.append(f"{flag} {config[key]}")
    return " ".join(args)


def write_csv(path: Path, rows: Sequence[PairedComparison]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _comparison_table(rows: Iterable[PairedComparison]) -> list[str]:
    lines = [
        "| Horizon | Method | Metric | Baseline | Candidate | Diff | 95% CI | "
        "Holm p | dz | W/L/T |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        holm = (
            row.holm_p_primary_family
            if row.holm_p_primary_family is not None
            else row.holm_p_all_reported
        )
        method = METHOD_LABELS.get(row.method, row.method)
        ci = f"[{row.diff_ci95_low:+.6f}, {row.diff_ci95_high:+.6f}]"
        wins = f"{row.wins}/{row.losses}/{row.ties}"
        lines.append(
            f"| {row.horizon_steps} | {method} | `{row.metric}` | "
            f"{row.baseline_mean:.6f} | {row.candidate_mean:.6f} | "
            f"{row.diff_mean_positive_favors_candidate:+.6f} | {ci} | "
            f"{_format_float(holm, 3)} | {row.paired_cohens_dz:+.3f} | {wins} |"
        )
    return lines


def _compute_table(payloads: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| Horizon | Method | Params | Extra state bytes | Train s | Steps/s | "
        "Slowdown vs FFN |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        steps = int(payload["config"]["steps"])
        grouped = _records_by_method(payload)
        baseline_speed = np.mean(
            [
                _summary_value(summary, "train_steps_per_s")
                for summary in grouped[BASELINE_METHOD].values()
            ]
        )
        for method, summaries in grouped.items():
            profile = payload["profiles"].get(method, {})
            train_s = np.mean(
                [_summary_value(summary, "train_s") for summary in summaries.values()]
            )
            speed = np.mean(
                [_summary_value(summary, "train_steps_per_s") for summary in summaries.values()]
            )
            slowdown = baseline_speed / speed if speed > 0 else float("nan")
            lines.append(
                f"| {steps} | {METHOD_LABELS.get(method, method)} | "
                f"{int(profile.get('trainable_params', 0))} | "
                f"{int(profile.get('state_bytes', 0))} | {train_s:.4f} | "
                f"{speed:.1f} | {slowdown:.2f}x |"
            )
    return lines


def _failure_mode_table(
    rows: Sequence[PairedComparison],
    payloads: Sequence[dict[str, Any]],
) -> list[str]:
    primary_lookup = {
        (row.horizon_steps, row.method, row.metric): row
        for row in rows
        if row.metric in DEFAULT_PRIMARY_METRICS
    }
    lines = [
        "| Horizon | Method | Open-gate negative advantage | Loses final NLL | "
        "Loses held-out NLL |",
        "|---:|---|---:|---:|---:|",
    ]
    for payload in payloads:
        steps = int(payload["config"]["steps"])
        grouped = _records_by_method(payload)
        for method in DEFAULT_CANDIDATE_METHODS:
            if method not in grouped:
                continue
            summaries = list(grouped[method].values())
            gates = np.asarray(
                [_summary_value(summary, "final_window_gate") for summary in summaries],
                dtype=np.float64,
            )
            advantages = np.asarray(
                [_summary_value(summary, "final_window_advantage") for summary in summaries],
                dtype=np.float64,
            )
            open_negative = bool(np.mean(gates) > 0.05 and np.mean(advantages) < 0.0)
            final_row = primary_lookup[(steps, method, "final_window_nll")]
            heldout_row = primary_lookup[(steps, method, "eval_nll")]
            lines.append(
                f"| {steps} | {METHOD_LABELS.get(method, method)} | "
                f"{open_negative} | "
                f"{final_row.diff_mean_positive_favors_candidate < 0.0} | "
                f"{heldout_row.diff_mean_positive_favors_candidate < 0.0} |"
            )
    return lines


def write_markdown(
    path: Path,
    *,
    manifest: dict[str, Any],
    rows: Sequence[PairedComparison],
    payloads: Sequence[dict[str, Any]],
    audit: Sequence[AuditCheck],
    primary_method: str,
    primary_metrics: Sequence[str],
) -> None:
    primary_rows = [
        row for row in rows if row.method == primary_method and row.metric in primary_metrics
    ]
    secondary_rows = [row for row in rows if row not in primary_rows]

    lines = [
        "# Step 2 Transformer Paperworthy Benchmark Report",
        "",
        "This report is generated from existing result artifacts; it is not a new training sweep.",
        "Positive paired differences favor the candidate method.",
        "",
        "## Inputs",
        "",
        f"Generated UTC: `{manifest['generated_at_utc']}`.",
        f"Git commit: `{manifest['git']['commit']}`.",
        f"Git dirty: `{manifest['git']['dirty']}`.",
        "",
        "| Result artifact | SHA256 | Steps | Seeds | Eval steps |",
        "|---|---|---:|---:|---:|",
    ]
    for source in manifest["sources"]:
        cfg = source["config"]
        lines.append(
            f"| `{source['path']}` | `{source['sha256'][:12]}` | "
            f"{cfg['steps']} | {cfg['seeds']} | {cfg['eval_steps']} |"
        )

    lines.extend(
        [
            "",
            "## Protocol Audit",
            "",
            "| Area | Status | Check | Evidence | Required action |",
            "|---|---|---|---|---|",
        ]
    )
    for check in audit:
        lines.append(
            f"| {check.area} | {check.status} | {check.check} | "
            f"{check.evidence} | {check.action} |"
        )

    lines.extend(
        [
            "",
            "## Primary Paired Results",
            "",
            "Family: "
            f"`{METHOD_LABELS.get(primary_method, primary_method)}` on "
            + ", ".join(f"`{metric}`" for metric in primary_metrics)
            + " across all input horizons. Holm p-values in this table correct "
            "within that primary family.",
            "",
            *_comparison_table(primary_rows),
            "",
            "## Secondary And Exploratory Results",
            "",
            "Holm p-values here correct across all reported comparisons, including "
            "diagnostic and compute metrics.",
            "",
            *_comparison_table(secondary_rows),
            "",
            "## Compute And Resource Reporting",
            "",
            *_compute_table(payloads),
            "",
            "## Failure Mode Flags",
            "",
            *_failure_mode_table(rows, payloads),
            "",
            "## Reproduction Commands From Captured Config",
            "",
        ]
    )
    for source in manifest["sources"]:
        lines.extend(["```bash", _command_from_config(source["config"]), "```", ""])

    lines.extend(
        [
            "## Remaining Paper Gaps",
            "",
            "- The current evidence is post hoc: the winning cap and replay settings were selected "
            "after exploratory sweeps.",
            "- Tiny Shakespeare is the only dataset, so corpus-specific effects are not ruled out.",
            "- The held-out split doubles as model-selection evidence; there is no "
            "untouched lockbox.",
            "- Ten paired seeds are too few for the observed sub-0.1% NLL margins.",
            "- Reproducibility fields are required schema; reject historical artifacts lacking "
            "per-seed offsets, exact command, git metadata, environment versions, or data hash.",
            "- Compute reporting lacks JIT compile separation, hardware identity, peak memory, and "
            "a parameter/compute-matched transformer baseline.",
            "- The canonical post-FFN memory still has a failure-mode smell: the gate can remain "
            "open while final-window measured advantage is negative.",
            "",
            "## Promotion Checklist",
            "",
            "- Freeze the data split, metric family, horizons, seeds, and primary "
            "method before running.",
            "- Run at least 30 paired seeds and report paired raw rows with exact stream offsets.",
            "- Use a validation split for tuning and a one-shot lockbox test split "
            "for the final claim.",
            "- Include parameter-matched, compute-matched, no-replay, no-cap, and "
            "wider-FFN baselines.",
            "- Report paired CIs, paired effect sizes, Holm-corrected p-values, "
            "win/loss/tie counts, "
            "and a practical minimum effect threshold.",
            "- Publish manifests with git status, command line, data sha256, "
            "package versions, device, "
            "timing warmup/hot-loop split, and resource counts.",
            "- Declare failure thresholds for NaNs, gate overuse, negative replay "
            "advantage, held-out "
            "regression, and seed outliers.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_csv_arg(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="*",
        default=list(DEFAULT_RESULT_PATHS),
        help="Result JSON paths to audit. Defaults to the three replay-capped 10-seed runs.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_CANDIDATE_METHODS),
        help="Comma-separated candidate methods to compare against the FFN baseline.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_REPORT_METRICS),
        help="Comma-separated metrics to report.",
    )
    parser.add_argument(
        "--primary-method",
        type=str,
        default=DEFAULT_PRIMARY_METHOD,
        help="Candidate method used for the confirmatory family.",
    )
    parser.add_argument(
        "--primary-metrics",
        type=str,
        default=",".join(DEFAULT_PRIMARY_METRICS),
        help="Comma-separated primary metrics for confirmatory Holm correction.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/benchmarks/step2_transformer_paperworthy_benchmark_suite"),
    )
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(path) for path in args.results]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    payloads = [_load_json(path) for path in paths]
    methods = _parse_csv_arg(args.methods) or tuple(_candidate_methods(payloads))
    metrics = _parse_csv_arg(args.metrics)
    primary_metrics = _parse_csv_arg(args.primary_metrics)

    rows: list[PairedComparison] = []
    for path, payload in zip(paths, payloads, strict=True):
        rows.extend(compare_payload(path, payload, methods=methods, metrics=metrics))
    rows = apply_corrections(
        rows,
        primary_method=args.primary_method,
        primary_metrics=primary_metrics,
        alpha=args.alpha,
    )
    manifest = build_manifest(paths, payloads)
    audit = audit_payloads(payloads)
    return {
        "manifest": manifest,
        "paired_comparisons": [asdict(row) for row in rows],
        "audit": [asdict(check) for check in audit],
        "settings": {
            "methods": list(methods),
            "metrics": list(metrics),
            "primary_method": args.primary_method,
            "primary_metrics": list(primary_metrics),
            "alpha": args.alpha,
        },
        "_rows": rows,
        "_payloads": payloads,
        "_audit_objects": audit,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = evaluate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = payload.pop("_rows")
    payloads = payload.pop("_payloads")
    audit = payload.pop("_audit_objects")
    manifest = payload["manifest"]

    write_json(args.output_dir / "config_manifest.json", manifest)
    write_csv(args.output_dir / "paired_stats.csv", rows)
    write_json(args.output_dir / "paperworthy_report.json", payload)
    write_markdown(
        args.output_dir / "paperworthy_report.md",
        manifest=manifest,
        rows=rows,
        payloads=payloads,
        audit=audit,
        primary_method=args.primary_method,
        primary_metrics=_parse_csv_arg(args.primary_metrics),
    )
    print(f"wrote {args.output_dir / 'config_manifest.json'}")
    print(f"wrote {args.output_dir / 'paired_stats.csv'}")
    print(f"wrote {args.output_dir / 'paperworthy_report.json'}")
    print(f"wrote {args.output_dir / 'paperworthy_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

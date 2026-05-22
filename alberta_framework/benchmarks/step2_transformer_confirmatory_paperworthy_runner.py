#!/usr/bin/env python3
"""Run the frozen Step 2 transformer confirmatory benchmark wrapper.

The wrapper keeps the main transformer runner unchanged. It materializes a
fixed validation or lockbox corpus, runs the replay-capped post-FFN candidate
suite through the existing runner, invokes the paperworthy report generator,
and writes a compact decision summary with preregistered failure flags.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

TRAIN_END_BYTE = 800_000
VALIDATION_END_BYTE = 950_000
PRIMARY_METHOD = "advantage_post_ffn_memory"
BASELINE_METHOD = "baseline_ffn_transformer"
PRIMARY_METRICS = ("final_window_nll", "eval_nll")
DEFAULT_HORIZONS = (3000, 5000, 10000)
SMOKE_HORIZONS = (300,)
DEFAULT_SEI_EVAL_NLL = 0.005
DEFAULT_ALPHA = 0.05
GATE_OPEN_THRESHOLD = 0.05
GATE_SATURATION_FRACTION = 0.90
ACTIVE_PROTOTYPE_SATURATION_FRACTION = 0.95


@dataclass(frozen=True)
class SplitArtifact:
    """Materialized corpus and split metadata."""

    preset: str
    source_path: str
    source_bytes: int
    source_sha256: str
    derived_path: str
    derived_bytes: int
    derived_sha256: str
    train_bytes: int
    train_sha256: str
    eval_bytes: int
    eval_sha256: str
    train_range: tuple[int, int]
    eval_range: tuple[int, int]
    train_fraction: str
    runner_split_index: int


@dataclass(frozen=True)
class ArtifactRecord:
    """Hashable file artifact metadata."""

    kind: str
    path: str
    exists: bool
    bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class CommandRecord:
    """One command executed or planned by this wrapper."""

    kind: str
    cwd: str
    command: list[str]
    output_dir: str | None = None
    skipped_existing: bool = False
    status: str = "planned"
    returncode: int | None = None
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    artifacts: tuple[ArtifactRecord, ...] = ()


def repo_root() -> Path:
    """Return the repository root from this benchmark script path."""
    return Path(__file__).resolve().parents[1]


def sha256_bytes(data: bytes) -> str:
    """Return a sha256 hex digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a sha256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, *, kind: str) -> ArtifactRecord:
    """Return existence, size, and sha256 metadata for one expected file."""
    exists = path.exists()
    if not exists or not path.is_file():
        return ArtifactRecord(kind=kind, path=str(path), exists=exists)
    return ArtifactRecord(
        kind=kind,
        path=str(path),
        exists=True,
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def run_git(root: Path, args: list[str]) -> str:
    """Run a git command and return stdout or ``unknown``."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def git_manifest(root: Path) -> dict[str, Any]:
    """Return git metadata for reproducibility."""
    status = run_git(root, ["status", "--porcelain"])
    return {
        "commit": run_git(root, ["rev-parse", "HEAD"]),
        "branch": run_git(root, ["branch", "--show-current"]),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def ensure_source_data(path: Path) -> bytes:
    """Read Tiny Shakespeare bytes, downloading them if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def split_fraction_string(train_len: int, total_len: int) -> tuple[str, int]:
    """Return a decimal train fraction whose runner split equals train_len."""
    base = train_len / total_len
    candidates = [base, np.nextafter(base, math.inf), np.nextafter(base, -math.inf)]
    for candidate in candidates:
        split = int(total_len * float(candidate))
        if split == train_len:
            return f"{float(candidate):.17g}", split
    for step in range(1, 100):
        candidate = base + step * 1e-16
        split = int(total_len * candidate)
        if split == train_len:
            return f"{candidate:.17g}", split
    raise RuntimeError("could not find stable train_fraction for derived corpus")


def materialize_split(
    *,
    preset: str,
    source_path: Path,
    output_root: Path,
) -> SplitArtifact:
    """Create the validation or lockbox corpus consumed by the unchanged runner."""
    if preset not in {"validation", "lockbox", "smoke"}:
        raise ValueError(f"unsupported preset for split materialization: {preset}")

    source = ensure_source_data(source_path)
    source.decode("utf-8")
    if len(source) <= VALIDATION_END_BYTE:
        raise ValueError("Tiny Shakespeare source is shorter than the frozen split")

    if preset == "lockbox":
        eval_start, eval_end = VALIDATION_END_BYTE, len(source)
    else:
        eval_start, eval_end = TRAIN_END_BYTE, VALIDATION_END_BYTE

    train = source[:TRAIN_END_BYTE]
    evaluation = source[eval_start:eval_end]
    derived = train + evaluation
    train_fraction, split_index = split_fraction_string(len(train), len(derived))
    if split_index != len(train):
        raise AssertionError("derived corpus split does not reproduce train range")

    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    derived_path = data_dir / f"tinyshakespeare_confirmatory_{preset}.txt"
    derived_path.write_bytes(derived)

    return SplitArtifact(
        preset=preset,
        source_path=str(source_path),
        source_bytes=len(source),
        source_sha256=sha256_bytes(source),
        derived_path=str(derived_path),
        derived_bytes=len(derived),
        derived_sha256=sha256_bytes(derived),
        train_bytes=len(train),
        train_sha256=sha256_bytes(train),
        eval_bytes=len(evaluation),
        eval_sha256=sha256_bytes(evaluation),
        train_range=(0, TRAIN_END_BYTE),
        eval_range=(eval_start, eval_end),
        train_fraction=train_fraction,
        runner_split_index=split_index,
    )


def shell_command(command: list[str]) -> str:
    """Return a copy-pastable shell command."""
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def runner_command(
    *,
    root: Path,
    split: SplitArtifact,
    output_dir: Path,
    steps: int,
    seeds: int,
    eval_steps: int,
    eval_batch_size: int,
    final_window: int,
    seed: int,
) -> list[str]:
    """Build one horizon command for the existing transformer runner."""
    script = (
        root
        / "examples/The Alberta Plan/Step2"
        / "step2_tiny_shakespeare_advantage_memory_transformer.py"
    )
    return [
        sys.executable,
        str(script),
        "--steps",
        str(steps),
        "--seeds",
        str(seeds),
        "--block-size",
        "32",
        "--d-model",
        "32",
        "--mlp-hidden",
        "64",
        "--proto-count",
        "64",
        "--eval-steps",
        str(eval_steps),
        "--eval-batch-size",
        str(eval_batch_size),
        "--final-window",
        str(final_window),
        "--train-fraction",
        split.train_fraction,
        "--baseline-lr",
        "0.15",
        "--fast-lr",
        "0.15",
        "--slow-lr",
        "0.1",
        "--grad-clip",
        "1.0",
        "--proto-update-rate",
        "0.3",
        "--proto-novelty-threshold",
        "0.0002",
        "--proto-bandwidth",
        "0.01",
        "--gate-init-logit",
        "-3.0",
        "--gate-lr",
        "0.5",
        "--gate-decay",
        "0.995",
        "--gate-max",
        "0.15",
        "--advantage-margin",
        "0.0",
        "--gate-l2",
        "0.1",
        "--gate-mode",
        "scalar",
        "--gate-objective",
        "replay",
        "--replay-size",
        "128",
        "--train-loss-mode",
        "memory",
        "--memory-loss-weight",
        "1.0",
        "--reset-mode",
        "meta_ema",
        "--seed",
        str(seed),
        "--data-path",
        split.derived_path,
        "--output-dir",
        str(output_dir),
    ]


def report_command(root: Path, result_paths: list[Path], output_dir: Path) -> list[str]:
    """Build the paperworthy report-generator command."""
    script = root / "benchmarks/step2_transformer_paperworthy_benchmark_suite.py"
    return [
        sys.executable,
        str(script),
        "--results",
        *[str(path) for path in result_paths],
        "--primary-method",
        PRIMARY_METHOD,
        "--primary-metrics",
        ",".join(PRIMARY_METRICS),
        "--output-dir",
        str(output_dir),
    ]


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(UTC).isoformat()


def run_command(
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
) -> tuple[str, int | None, str | None, str | None]:
    """Run a command unless this is a dry run, returning manifest status fields."""
    print(shell_command(command))
    if dry_run:
        return "planned", None, None, None
    started_at = utc_now()
    completed = subprocess.run(command, cwd=cwd, check=False)
    completed_at = utc_now()
    status = "completed" if completed.returncode == 0 else "failed"
    return status, completed.returncode, started_at, completed_at


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object in {path}")
    return parsed


def validate_lockbox_access(args: argparse.Namespace) -> dict[str, Any]:
    """Return lockbox gate metadata or raise before any lockbox split access."""
    gate: dict[str, Any] = {
        "lockbox_requested": args.preset == "lockbox",
        "validation_decision_summary_required": args.preset == "lockbox",
        "validation_decision_summary_path": None,
        "validation_clears_confirmatory_bar": None,
        "allow_lockbox_without_validation": bool(args.allow_lockbox_without_validation),
        "dry_run": bool(args.dry_run),
        "status": "not_required",
        "never_evaluates": bool(args.dry_run),
    }
    if args.preset != "lockbox":
        return gate

    validation_path = args.validation_decision_summary
    if validation_path is not None:
        gate["validation_decision_summary_path"] = str(validation_path)
        gate["validation_decision_summary_artifact"] = asdict(
            artifact_record(validation_path, kind="validation_decision_summary")
        )
        if not validation_path.exists():
            gate["status"] = "validation_summary_missing"
            raise ValueError(
                "--preset lockbox requires an existing --validation-decision-summary"
            )
        summary = load_json(validation_path)
        clears_bar = summary.get("clears_confirmatory_bar") is True
        gate["validation_clears_confirmatory_bar"] = clears_bar
        gate["validation_decision_schema"] = summary.get("schema")
        gate["status"] = "validation_cleared" if clears_bar else "validation_failed"
        if not clears_bar:
            raise ValueError(
                "--preset lockbox requires a validation decision summary with "
                "clears_confirmatory_bar exactly true"
            )
        return gate

    if args.allow_lockbox_without_validation:
        if not args.dry_run:
            gate["status"] = "planning_mode_requires_dry_run"
            raise ValueError(
                "--allow-lockbox-without-validation is only valid with --dry-run"
            )
        gate["status"] = "dry_run_planning_without_validation"
        gate["never_evaluates"] = True
        return gate

    gate["status"] = "validation_summary_required"
    raise ValueError(
        "--preset lockbox requires --validation-decision-summary pointing to a "
        "validation confirmatory_decision_summary.json with "
        "clears_confirmatory_bar true. For command planning only, pass both "
        "--dry-run and --allow-lockbox-without-validation."
    )


def finite_metric_failures(result_paths: list[Path]) -> list[dict[str, Any]]:
    """Return records with non-finite metric values."""
    failures: list[dict[str, Any]] = []
    for path in result_paths:
        payload = load_json(path)
        for record in payload["records"]:
            for metric, value in record["summary"].items():
                numeric = float(value)
                if not math.isfinite(numeric):
                    failures.append(
                        {
                            "path": str(path),
                            "seed": int(record["seed"]),
                            "method": str(record["method"]),
                            "metric": str(metric),
                            "value": numeric,
                        }
                    )
    return failures


def primary_rows(report_dir: Path) -> list[dict[str, Any]]:
    """Load primary paired-comparison rows from the generated CSV."""
    path = report_dir / "paired_stats.csv"
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == PRIMARY_METHOD and row["metric"] in PRIMARY_METRICS:
                rows.append(dict(row))
    return rows


def rows_by_seed(payload: dict[str, Any], metric: str) -> np.ndarray:
    """Return paired candidate-favoring diffs by seed for one metric."""
    baseline = {
        int(record["seed"]): float(record["summary"][metric])
        for record in payload["records"]
        if record["method"] == BASELINE_METHOD
    }
    candidate = {
        int(record["seed"]): float(record["summary"][metric])
        for record in payload["records"]
        if record["method"] == PRIMARY_METHOD
    }
    seeds = sorted(set(baseline).intersection(candidate))
    if metric in {"final_window_nll", "eval_nll"}:
        return np.asarray(
            [baseline[seed] - candidate[seed] for seed in seeds],
            dtype=np.float64,
        )
    return np.asarray(
        [candidate[seed] - baseline[seed] for seed in seeds],
        dtype=np.float64,
    )


def outlier_flags(result_paths: list[Path]) -> list[dict[str, Any]]:
    """Return paired seed outliers beyond three sample standard deviations."""
    flags: list[dict[str, Any]] = []
    for path in result_paths:
        payload = load_json(path)
        steps = int(payload["config"]["steps"])
        for metric in PRIMARY_METRICS:
            diffs = rows_by_seed(payload, metric)
            if diffs.size < 3:
                continue
            std = float(np.std(diffs, ddof=1))
            if std == 0.0:
                continue
            mean = float(np.mean(diffs))
            for idx, value in enumerate(diffs):
                z_score = (float(value) - mean) / std
                if abs(z_score) > 3.0:
                    flags.append(
                        {
                            "path": str(path),
                            "horizon_steps": steps,
                            "metric": metric,
                            "paired_index": idx,
                            "diff": float(value),
                            "z_score": z_score,
                        }
                    )
    return flags


def mechanism_flags(result_paths: list[Path]) -> list[dict[str, Any]]:
    """Return gate/advantage/resource failure flags for primary method records."""
    flags: list[dict[str, Any]] = []
    for path in result_paths:
        payload = load_json(path)
        config = payload["config"]
        gate_max = float(config["gate_max"])
        proto_count = float(config["proto_count"])
        records = [
            record
            for record in payload["records"]
            if record["method"] == PRIMARY_METHOD
        ]
        if not records:
            continue
        gates = np.asarray(
            [float(record["summary"]["final_window_gate"]) for record in records],
            dtype=np.float64,
        )
        advantages = np.asarray(
            [float(record["summary"]["final_window_advantage"]) for record in records],
            dtype=np.float64,
        )
        active = np.asarray(
            [float(record["summary"]["final_window_active_prototypes"]) for record in records],
            dtype=np.float64,
        )
        mean_gate = float(np.mean(gates))
        mean_advantage = float(np.mean(advantages))
        mean_active = float(np.mean(active))
        if mean_gate > GATE_OPEN_THRESHOLD and mean_advantage < 0.0:
            flags.append(
                {
                    "path": str(path),
                    "horizon_steps": int(config["steps"]),
                    "flag": "open_gate_negative_advantage",
                    "mean_gate": mean_gate,
                    "mean_advantage": mean_advantage,
                }
            )
        if mean_gate >= GATE_SATURATION_FRACTION * gate_max:
            flags.append(
                {
                    "path": str(path),
                    "horizon_steps": int(config["steps"]),
                    "flag": "gate_saturation",
                    "mean_gate": mean_gate,
                    "gate_max": gate_max,
                }
            )
        if mean_active >= ACTIVE_PROTOTYPE_SATURATION_FRACTION * proto_count:
            flags.append(
                {
                    "path": str(path),
                    "horizon_steps": int(config["steps"]),
                    "flag": "active_prototype_saturation",
                    "mean_active_prototypes": mean_active,
                    "proto_count": proto_count,
                }
            )
    return flags


REQUIRED_OFFSET_KEYS = (
    "train_offset",
    "train_effective_offset",
    "eval_offset",
    "eval_effective_offset",
)


def nested_offset_failures(result_paths: list[Path]) -> list[dict[str, Any]]:
    """Return records missing replayable per-seed train/eval offset metadata."""
    failures: list[dict[str, Any]] = []
    for path in result_paths:
        payload = load_json(path)
        manifest = payload.get("manifest", {})
        seed_runs_raw = manifest.get("seed_runs", []) if isinstance(manifest, dict) else []
        seed_runs = {
            int(seed_run["seed_index"]): seed_run
            for seed_run in seed_runs_raw
            if isinstance(seed_run, dict) and "seed_index" in seed_run
        }
        if not seed_runs:
            failures.append(
                {
                    "path": str(path),
                    "scope": "manifest.seed_runs",
                    "missing": ["seed_runs"],
                }
            )
        for record_index, record in enumerate(payload["records"]):
            seed = int(record["seed"])
            method = str(record["method"])
            data_offsets = record.get("data_offsets")
            if not isinstance(data_offsets, dict):
                failures.append(
                    {
                        "path": str(path),
                        "record_index": record_index,
                        "seed": seed,
                        "method": method,
                        "scope": "record.data_offsets",
                        "missing": list(REQUIRED_OFFSET_KEYS),
                    }
                )
                data_offsets = {}
            missing_record_keys = [
                key for key in REQUIRED_OFFSET_KEYS if key not in data_offsets
            ]
            if missing_record_keys:
                failures.append(
                    {
                        "path": str(path),
                        "record_index": record_index,
                        "seed": seed,
                        "method": method,
                        "scope": "record.data_offsets",
                        "missing": missing_record_keys,
                    }
                )

            seed_run = seed_runs.get(seed)
            methods = seed_run.get("methods", {}) if isinstance(seed_run, dict) else {}
            method_offsets = methods.get(method) if isinstance(methods, dict) else None
            if not isinstance(method_offsets, dict):
                failures.append(
                    {
                        "path": str(path),
                        "record_index": record_index,
                        "seed": seed,
                        "method": method,
                        "scope": "manifest.seed_runs.methods",
                        "missing": list(REQUIRED_OFFSET_KEYS),
                    }
                )
                continue
            missing_manifest_keys = [
                key for key in REQUIRED_OFFSET_KEYS if key not in method_offsets
            ]
            if missing_manifest_keys:
                failures.append(
                    {
                        "path": str(path),
                        "record_index": record_index,
                        "seed": seed,
                        "method": method,
                        "scope": "manifest.seed_runs.methods",
                        "missing": missing_manifest_keys,
                    }
                )
            mismatched_keys = [
                key
                for key in REQUIRED_OFFSET_KEYS
                if key in data_offsets
                and key in method_offsets
                and int(data_offsets[key]) != int(method_offsets[key])
            ]
            if mismatched_keys:
                failures.append(
                    {
                        "path": str(path),
                        "record_index": record_index,
                        "seed": seed,
                        "method": method,
                        "scope": "offset_consistency",
                        "mismatched": mismatched_keys,
                    }
                )
    return failures


def summarize_decision(
    *,
    result_paths: list[Path],
    report_dir: Path,
    seeds: int,
    expected_seeds: int,
    main_horizon: int,
    sei_eval_nll: float,
    alpha: float,
) -> dict[str, Any]:
    """Summarize the preregistered confirmatory flags."""
    rows = primary_rows(report_dir)
    nonfinite = finite_metric_failures(result_paths)
    negative_primary = [
        row
        for row in rows
        if float(row["diff_mean_positive_favors_candidate"]) <= 0.0
    ]
    holm_fail = [
        row
        for row in rows
        if row["holm_p_primary_family"]
        and float(row["holm_p_primary_family"]) >= alpha
    ]
    underpowered = seeds < expected_seeds
    main_eval_rows = [
        row
        for row in rows
        if int(row["horizon_steps"]) == main_horizon and row["metric"] == "eval_nll"
    ]
    sei_fail = not main_eval_rows or float(main_eval_rows[0]["diff_ci95_low"]) < sei_eval_nll
    offset_failures = nested_offset_failures(result_paths)
    flags = {
        "nonfinite_metrics": nonfinite,
        "negative_primary_means": negative_primary,
        "holm_primary_not_significant": holm_fail,
        "underpowered_seed_count": underpowered,
        "main_horizon_eval_nll_ci_clears_sei": not sei_fail,
        "mechanism_flags": mechanism_flags(result_paths),
        "seed_outliers": outlier_flags(result_paths),
        "nested_per_seed_offsets_present": not offset_failures,
        "missing_nested_per_seed_offsets": offset_failures,
    }
    clears_bar = (
        not nonfinite
        and not negative_primary
        and not holm_fail
        and not underpowered
        and not sei_fail
        and not flags["mechanism_flags"]
        and not offset_failures
    )
    return {
        "schema": "alberta.step2.transformer_confirmatory_decision.v2",
        "generated_at_utc": utc_now(),
        "primary_method": PRIMARY_METHOD,
        "primary_metrics": list(PRIMARY_METRICS),
        "main_horizon": main_horizon,
        "smallest_effect_of_interest_eval_nll": sei_eval_nll,
        "alpha": alpha,
        "clears_confirmatory_bar": clears_bar,
        "flags": flags,
        "primary_rows": rows,
    }


def write_summary_markdown(path: Path, summary: dict[str, Any], report_dir: Path) -> None:
    """Write a compact human-readable decision summary."""
    lines = [
        "# Step 2 Transformer Confirmatory Summary",
        "",
        f"Generated UTC: `{summary['generated_at_utc']}`.",
        f"Primary method: `{summary['primary_method']}`.",
        f"Clears confirmatory bar: `{summary['clears_confirmatory_bar']}`.",
        "",
        "## Primary Rows",
        "",
        "| Horizon | Metric | N | Diff | 95% CI | Holm p | W/L/T |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["primary_rows"]:
        ci = f"[{float(row['diff_ci95_low']):+.6f}, {float(row['diff_ci95_high']):+.6f}]"
        wins = f"{row['wins']}/{row['losses']}/{row['ties']}"
        lines.append(
            f"| {row['horizon_steps']} | `{row['metric']}` | {row['n_pairs']} | "
            f"{float(row['diff_mean_positive_favors_candidate']):+.6f} | "
            f"{ci} | {float(row['holm_p_primary_family']):.6g} | {wins} |"
        )
    lines.extend(
        [
            "",
            "## Flags",
            "",
            f"- nonfinite metrics: `{len(summary['flags']['nonfinite_metrics'])}`",
            f"- negative primary means: `{len(summary['flags']['negative_primary_means'])}`",
            "- Holm primary not significant: "
            f"`{len(summary['flags']['holm_primary_not_significant'])}`",
            f"- underpowered seed count: `{summary['flags']['underpowered_seed_count']}`",
            "- main-horizon held-out CI clears SEI: "
            f"`{summary['flags']['main_horizon_eval_nll_ci_clears_sei']}`",
            f"- mechanism flags: `{len(summary['flags']['mechanism_flags'])}`",
            f"- seed outliers: `{len(summary['flags']['seed_outliers'])}`",
            "- nested per-seed offsets present: "
            f"`{summary['flags']['nested_per_seed_offsets_present']}`",
            "- missing nested per-seed offset records: "
            f"`{len(summary['flags']['missing_nested_per_seed_offsets'])}`",
            "",
            "## Report Artifacts",
            "",
            f"- `{report_dir / 'paired_stats.csv'}`",
            f"- `{report_dir / 'paperworthy_report.md'}`",
            f"- `{report_dir / 'paperworthy_report.json'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def preset_defaults(preset: str) -> tuple[tuple[int, ...], int, int, int, int]:
    """Return horizons, seeds, eval steps, eval batch size, and final window."""
    if preset == "smoke":
        return SMOKE_HORIZONS, 2, 128, 128, 128
    return DEFAULT_HORIZONS, 30, 4096, 512, 512


def effective_eval_batch_size(eval_steps: int, eval_batch_size: int) -> int:
    """Return the held-out batch size passed to the transformer runner."""
    if eval_batch_size <= 0:
        return eval_steps
    return min(eval_steps, eval_batch_size)


def eval_batch_count(eval_steps: int, eval_batch_size: int) -> int:
    """Return held-out evaluation batches per seed."""
    effective = effective_eval_batch_size(eval_steps, eval_batch_size)
    return (eval_steps + effective - 1) // effective


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "validation", "lockbox"),
        default="smoke",
        help="Run a small smoke, the frozen validation suite, or the one-shot lockbox suite.",
    )
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help=(
            "Held-out eval batch size passed to the transformer runner. "
            "Use 0 for legacy full-context eval."
        ),
    )
    parser.add_argument("--final-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source-data-path",
        type=Path,
        default=Path("output/subagents/transformer_ffn/data/tinyshakespeare.txt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/step2_new_directions/advantage_memory_transformer_confirmatory_smoke"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun horizons with existing results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and manifest commands without running.",
    )
    parser.add_argument(
        "--validation-decision-summary",
        "--validation-decision-summary-path",
        dest="validation_decision_summary",
        type=Path,
        default=None,
        help=(
            "Required for --preset lockbox unless planning with --dry-run and "
            "--allow-lockbox-without-validation. The JSON must contain "
            "clears_confirmatory_bar: true."
        ),
    )
    parser.add_argument(
        "--allow-lockbox-without-validation",
        action="store_true",
        help=(
            "Allow --preset lockbox only as a dry-run command-planning path "
            "without a validation decision summary. This mode never evaluates."
        ),
    )
    parser.add_argument("--expected-seeds", type=int, default=30)
    parser.add_argument("--sei-eval-nll", type=float, default=DEFAULT_SEI_EVAL_NLL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser.parse_args()


def report_output_artifacts(report_dir: Path) -> tuple[ArtifactRecord, ...]:
    """Return expected paperworthy report outputs."""
    return (
        artifact_record(report_dir / "config_manifest.json", kind="report_output"),
        artifact_record(report_dir / "paired_stats.csv", kind="report_output"),
        artifact_record(report_dir / "paperworthy_report.json", kind="report_output"),
        artifact_record(report_dir / "paperworthy_report.md", kind="report_output"),
    )


def build_wrapper_manifest(
    *,
    args: argparse.Namespace,
    root: Path,
    split: SplitArtifact,
    steps: tuple[int, ...],
    seeds: int,
    eval_steps: int,
    eval_batch_size: int,
    effective_batch_size: int,
    final_window: int,
    command_records: list[CommandRecord],
    result_paths: list[Path],
    report_dir: Path,
    decision_output_paths: tuple[Path, ...],
    lockbox_gate: dict[str, Any],
) -> dict[str, Any]:
    """Build the wrapper manifest with reproducibility hashes and command status."""
    return {
        "schema": "alberta.step2.transformer_confirmatory_wrapper.v2",
        "generated_at_utc": utc_now(),
        "preset": args.preset,
        "lockbox_gate": lockbox_gate,
        "protocol": {
            "primary_method": PRIMARY_METHOD,
            "primary_metrics": list(PRIMARY_METRICS),
            "horizons": list(steps),
            "seeds": seeds,
            "eval_steps": eval_steps,
            "eval_batch_size": effective_batch_size,
            "requested_eval_batch_size": eval_batch_size,
            "eval_batches_per_seed": eval_batch_count(eval_steps, eval_batch_size),
            "eval_aggregation": "weighted mean over all held-out examples; no subsampling",
            "final_window": final_window,
            "smallest_effect_of_interest_eval_nll": args.sei_eval_nll,
            "alpha": args.alpha,
        },
        "split": asdict(split),
        "artifacts": {
            "data_splits": {
                "source": asdict(
                    artifact_record(Path(split.source_path), kind="source_corpus")
                ),
                "train": {
                    "byte_range": list(split.train_range),
                    "bytes": split.train_bytes,
                    "sha256": split.train_sha256,
                },
                "eval": {
                    "byte_range": list(split.eval_range),
                    "bytes": split.eval_bytes,
                    "sha256": split.eval_sha256,
                },
                "derived": asdict(
                    artifact_record(Path(split.derived_path), kind="derived_corpus")
                ),
            },
            "runner_results": [
                asdict(artifact_record(path, kind="runner_result"))
                for path in result_paths
            ],
            "report_inputs": [
                asdict(artifact_record(path, kind="report_input"))
                for path in result_paths
            ],
            "report_outputs": [
                asdict(record) for record in report_output_artifacts(report_dir)
            ],
            "decision_outputs": [
                asdict(artifact_record(path, kind="decision_output"))
                for path in decision_output_paths
            ],
        },
        "commands": [asdict(record) for record in command_records],
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "git": git_manifest(root),
    }


def write_wrapper_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    root: Path,
    split: SplitArtifact,
    steps: tuple[int, ...],
    seeds: int,
    eval_steps: int,
    eval_batch_size: int,
    effective_batch_size: int,
    final_window: int,
    command_records: list[CommandRecord],
    result_paths: list[Path],
    report_dir: Path,
    decision_output_paths: tuple[Path, ...],
    lockbox_gate: dict[str, Any],
) -> None:
    """Write the wrapper manifest."""
    manifest = build_wrapper_manifest(
        args=args,
        root=root,
        split=split,
        steps=steps,
        seeds=seeds,
        eval_steps=eval_steps,
        eval_batch_size=eval_batch_size,
        effective_batch_size=effective_batch_size,
        final_window=final_window,
        command_records=command_records,
        result_paths=result_paths,
        report_dir=report_dir,
        decision_output_paths=decision_output_paths,
        lockbox_gate=lockbox_gate,
    )
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    """Run or plan the confirmatory benchmark suite."""
    args = parse_args()
    try:
        lockbox_gate = validate_lockbox_access(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    (
        defaults_steps,
        defaults_seeds,
        defaults_eval,
        defaults_eval_batch_size,
        defaults_final,
    ) = preset_defaults(args.preset)
    steps = tuple(args.steps) if args.steps else defaults_steps
    seeds = args.seeds if args.seeds is not None else defaults_seeds
    eval_steps = args.eval_steps if args.eval_steps is not None else defaults_eval
    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size is not None
        else defaults_eval_batch_size
    )
    if eval_batch_size < 0:
        raise ValueError("--eval-batch-size must be non-negative")
    effective_batch_size = effective_eval_batch_size(eval_steps, eval_batch_size)
    final_window = args.final_window if args.final_window is not None else defaults_final

    root = repo_root()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "confirmatory_wrapper_manifest.json"
    report_dir = output_root / f"{args.preset}_paperworthy_report"
    split = materialize_split(
        preset=args.preset,
        source_path=args.source_data_path,
        output_root=output_root,
    )

    command_records: list[CommandRecord] = []
    result_paths: list[Path] = []
    for horizon in steps:
        run_dir = output_root / (
            f"{args.preset}_{horizon}_{seeds}seed_eval{eval_steps}_fw{final_window}_"
            f"eb{effective_batch_size}_replay128_scalar_glr05_l2_01_gmax015"
        )
        result_path = run_dir / "results.json"
        result_paths.append(result_path)
        command = runner_command(
            root=root,
            split=split,
            output_dir=run_dir,
            steps=horizon,
            seeds=seeds,
            eval_steps=eval_steps,
            eval_batch_size=eval_batch_size,
            final_window=final_window,
            seed=args.seed,
        )
        skipped = result_path.exists() and not args.force
        record = CommandRecord(
            kind="runner",
            cwd=str(root),
            command=command,
            output_dir=str(run_dir),
            skipped_existing=skipped,
        )
        if skipped:
            print(f"skipping existing {result_path}")
            command_records.append(
                replace(
                    record,
                    status="skipped_existing",
                    returncode=0,
                    artifacts=(artifact_record(result_path, kind="runner_result"),),
                )
            )
            continue
        status, returncode, started_at, completed_at = run_command(
            command,
            cwd=root,
            dry_run=args.dry_run,
        )
        command_records.append(
            replace(
                record,
                status=status,
                returncode=returncode,
                started_at_utc=started_at,
                completed_at_utc=completed_at,
                artifacts=(artifact_record(result_path, kind="runner_result"),),
            )
        )
        if status == "failed":
            write_wrapper_manifest(
                manifest_path,
                args=args,
                root=root,
                split=split,
                steps=steps,
                seeds=seeds,
                eval_steps=eval_steps,
                eval_batch_size=eval_batch_size,
                effective_batch_size=effective_batch_size,
                final_window=final_window,
                command_records=command_records,
                result_paths=result_paths,
                report_dir=report_dir,
                decision_output_paths=(),
                lockbox_gate=lockbox_gate,
            )
            print(f"wrote {manifest_path}")
            return returncode or 1

    report = report_command(root, result_paths, report_dir)
    report_record = CommandRecord(
        kind="paperworthy_report",
        cwd=str(root),
        command=report,
        output_dir=str(report_dir),
        skipped_existing=False,
    )

    if args.dry_run:
        print(shell_command(report))
        command_records.append(
            replace(
                report_record,
                status="planned",
                artifacts=report_output_artifacts(report_dir),
            )
        )
        write_wrapper_manifest(
            manifest_path,
            args=args,
            root=root,
            split=split,
            steps=steps,
            seeds=seeds,
            eval_steps=eval_steps,
            eval_batch_size=eval_batch_size,
            effective_batch_size=effective_batch_size,
            final_window=final_window,
            command_records=command_records,
            result_paths=result_paths,
            report_dir=report_dir,
            decision_output_paths=(),
            lockbox_gate=lockbox_gate,
        )
        print(f"wrote {manifest_path}")
        return 0

    status, returncode, started_at, completed_at = run_command(
        report,
        cwd=root,
        dry_run=False,
    )
    command_records.append(
        replace(
            report_record,
            status=status,
            returncode=returncode,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            artifacts=report_output_artifacts(report_dir),
        )
    )
    if status == "failed":
        write_wrapper_manifest(
            manifest_path,
            args=args,
            root=root,
            split=split,
            steps=steps,
            seeds=seeds,
            eval_steps=eval_steps,
            eval_batch_size=eval_batch_size,
            effective_batch_size=effective_batch_size,
            final_window=final_window,
            command_records=command_records,
            result_paths=result_paths,
            report_dir=report_dir,
            decision_output_paths=(),
            lockbox_gate=lockbox_gate,
        )
        print(f"wrote {manifest_path}")
        return returncode or 1

    summary = summarize_decision(
        result_paths=result_paths,
        report_dir=report_dir,
        seeds=seeds,
        expected_seeds=args.expected_seeds,
        main_horizon=max(steps),
        sei_eval_nll=args.sei_eval_nll,
        alpha=args.alpha,
    )
    summary_path = output_root / "confirmatory_decision_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(
        output_root / "confirmatory_decision_summary.md",
        summary,
        report_dir,
    )
    write_wrapper_manifest(
        manifest_path,
        args=args,
        root=root,
        split=split,
        steps=steps,
        seeds=seeds,
        eval_steps=eval_steps,
        eval_batch_size=eval_batch_size,
        effective_batch_size=effective_batch_size,
        final_window=final_window,
        command_records=command_records,
        result_paths=result_paths,
        report_dir=report_dir,
        decision_output_paths=(
            summary_path,
            output_root / "confirmatory_decision_summary.md",
        ),
        lockbox_gate=lockbox_gate,
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {output_root / 'confirmatory_decision_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

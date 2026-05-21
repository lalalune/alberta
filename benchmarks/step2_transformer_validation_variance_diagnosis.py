#!/usr/bin/env python3
"""Diagnose the completed Step 2 transformer validation failure.

This script is analysis-only. It reads validation `results.json` artifacts and
does not touch the lockbox split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

DEFAULT_RESULT_ROOT = Path(
    "outputs/step2_new_directions/"
    "advantage_memory_transformer_confirmatory_validation_30seed"
)
DEFAULT_RESULT_GLOB = (
    "validation_*_30seed_eval4096_fw512_eb512_replay128_scalar_glr05_l2_01_gmax015/"
    "results.json"
)
BASELINE_METHOD = "baseline_ffn_transformer"
METHODS = ("advantage_post_ffn_memory", "advantage_pre_ffn_kv_memory")
PRIMARY_METRICS = ("final_window_nll", "eval_nll")
SEI_EVAL_NLL = 0.005
Z95 = 1.96


@dataclass(frozen=True)
class DiffRow:
    """One paired-difference diagnostic row."""

    source: str
    horizon_steps: int
    method: str
    metric: str
    n: int
    mean_diff_positive_favors_method: float
    stderr: float
    ci95_low: float
    ci95_high: float
    wins: int
    losses: int
    ties: int
    paired_sd: float
    required_n_for_ci_low_above_zero: int | None
    required_n_for_eval_sei: int | None


@dataclass(frozen=True)
class CorrelationRow:
    """Correlation between two paired-difference series."""

    source: str
    horizon_steps: int
    method: str
    x_metric: str
    y_metric: str
    pearson_r: float | None


@dataclass(frozen=True)
class MechanismRow:
    """Mechanism diagnostic row."""

    source: str
    horizon_steps: int
    method: str
    final_window_gate_mean: float | None
    final_window_advantage_mean: float | None
    final_window_active_prototypes_mean: float | None
    open_gate_negative_advantage: bool
    eval_improves_but_final_window_does_not: bool
    eval_fast_diff: float | None


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object in {path}")
    return parsed


def float_summary(record: dict[str, Any], metric: str) -> float:
    """Return a metric from a record summary."""
    summary = cast(dict[str, Any], record["summary"])
    return float(cast(float | int, summary[metric]))


def records_by_method(payload: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    """Group records by method and seed."""
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for record in cast(list[dict[str, Any]], payload["records"]):
        grouped.setdefault(str(record["method"]), {})[int(record["seed"])] = record
    return grouped


def paired_diffs(
    payload: dict[str, Any],
    *,
    method: str,
    metric: str,
) -> np.ndarray:
    """Return paired baseline minus method diffs for lower-is-better metrics."""
    baseline_metric = "eval_nll" if metric == "eval_fast_nll" else metric
    grouped = records_by_method(payload)
    baseline = grouped[BASELINE_METHOD]
    candidate = grouped[method]
    seeds = sorted(set(baseline).intersection(candidate))
    diffs = np.asarray(
        [
            float_summary(baseline[seed], baseline_metric)
            - float_summary(candidate[seed], metric)
            for seed in seeds
        ],
        dtype=np.float64,
    )
    return cast(np.ndarray, diffs)


def required_n_for_ci(
    *,
    mean: float,
    paired_sd: float,
    threshold: float,
) -> int | None:
    """Approximate paired n needed for a 95% lower CI above threshold."""
    margin = mean - threshold
    if margin <= 0.0 or paired_sd <= 0.0:
        return None
    return int(math.ceil((Z95 * paired_sd / margin) ** 2))


def diff_row(path: Path, payload: dict[str, Any], method: str, metric: str) -> DiffRow:
    """Build one paired-diff row."""
    diffs = paired_diffs(payload, method=method, metric=metric)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
    stderr = sd / math.sqrt(int(diffs.size)) if diffs.size > 1 else 0.0
    config = cast(dict[str, Any], payload["config"])
    return DiffRow(
        source=str(path),
        horizon_steps=int(config["steps"]),
        method=method,
        metric=metric,
        n=int(diffs.size),
        mean_diff_positive_favors_method=mean,
        stderr=stderr,
        ci95_low=mean - Z95 * stderr,
        ci95_high=mean + Z95 * stderr,
        wins=int(np.sum(diffs > 0.0)),
        losses=int(np.sum(diffs < 0.0)),
        ties=int(np.sum(diffs == 0.0)),
        paired_sd=sd,
        required_n_for_ci_low_above_zero=required_n_for_ci(
            mean=mean,
            paired_sd=sd,
            threshold=0.0,
        ),
        required_n_for_eval_sei=required_n_for_ci(
            mean=mean,
            paired_sd=sd,
            threshold=SEI_EVAL_NLL,
        )
        if metric == "eval_nll"
        else None,
    )


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return Pearson correlation or None when undefined."""
    if x.size < 3 or y.size < 3:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlation_rows(path: Path, payload: dict[str, Any], method: str) -> list[CorrelationRow]:
    """Build correlation diagnostics for one method/horizon."""
    config = cast(dict[str, Any], payload["config"])
    horizon = int(config["steps"])
    final = paired_diffs(payload, method=method, metric="final_window_nll")
    eval_nll = paired_diffs(payload, method=method, metric="eval_nll")
    eval_fast = paired_diffs(payload, method=method, metric="eval_fast_nll")
    rows = [
        CorrelationRow(
            source=str(path),
            horizon_steps=horizon,
            method=method,
            x_metric="final_window_nll",
            y_metric="eval_nll",
            pearson_r=pearson(final, eval_nll),
        ),
        CorrelationRow(
            source=str(path),
            horizon_steps=horizon,
            method=method,
            x_metric="eval_nll",
            y_metric="eval_fast_nll",
            pearson_r=pearson(eval_nll, eval_fast),
        ),
    ]
    grouped = records_by_method(payload)
    offsets = []
    for seed, record in sorted(grouped[method].items()):
        data_offsets = cast(dict[str, Any], record.get("data_offsets", {}))
        offsets.append(float(data_offsets.get("eval_effective_offset", seed)))
    rows.append(
        CorrelationRow(
            source=str(path),
            horizon_steps=horizon,
            method=method,
            x_metric="eval_effective_offset",
            y_metric="eval_nll",
            pearson_r=pearson(np.asarray(offsets, dtype=np.float64), eval_nll),
        )
    )
    return rows


def maybe_mean(payload: dict[str, Any], method: str, metric: str) -> float | None:
    """Return mean metric for method if present."""
    values = [
        float_summary(record, metric)
        for record in records_by_method(payload).get(method, {}).values()
        if metric in cast(dict[str, Any], record["summary"])
    ]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def mechanism_row(path: Path, payload: dict[str, Any], method: str) -> MechanismRow:
    """Build one mechanism row."""
    config = cast(dict[str, Any], payload["config"])
    final_row = diff_row(path, payload, method, "final_window_nll")
    eval_row = diff_row(path, payload, method, "eval_nll")
    eval_fast = diff_row(path, payload, method, "eval_fast_nll")
    gate = maybe_mean(payload, method, "final_window_gate")
    advantage = maybe_mean(payload, method, "final_window_advantage")
    active = maybe_mean(payload, method, "final_window_active_prototypes")
    return MechanismRow(
        source=str(path),
        horizon_steps=int(config["steps"]),
        method=method,
        final_window_gate_mean=gate,
        final_window_advantage_mean=advantage,
        final_window_active_prototypes_mean=active,
        open_gate_negative_advantage=bool(
            gate is not None and advantage is not None and gate > 0.05 and advantage < 0.0
        ),
        eval_improves_but_final_window_does_not=bool(
            eval_row.ci95_low > 0.0 and final_row.ci95_low <= 0.0
        ),
        eval_fast_diff=eval_fast.mean_diff_positive_favors_method,
    )


def write_csv(path: Path, rows: list[Any]) -> None:
    """Write dataclass rows to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def fmt_optional(value: float | int | None, digits: int = 6) -> str:
    """Format optional numeric values."""
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_markdown(
    path: Path,
    *,
    diff_rows: list[DiffRow],
    corr_rows: list[CorrelationRow],
    mechanism_rows: list[MechanismRow],
) -> None:
    """Write human-readable diagnosis."""
    lines = [
        "# Step 2 Transformer Validation Variance Diagnosis",
        "",
        "This report reads validation artifacts only. It does not access lockbox.",
        "",
        "## Primary Diff Rows",
        "",
        "| Horizon | Method | Metric | Diff | 95% CI | W/L/T | n for CI>0 | n for eval SEI |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for diff in sorted(diff_rows, key=lambda item: (item.horizon_steps, item.method, item.metric)):
        ci = f"[{diff.ci95_low:+.6f}, {diff.ci95_high:+.6f}]"
        wins = f"{diff.wins}/{diff.losses}/{diff.ties}"
        lines.append(
            f"| {diff.horizon_steps} | `{diff.method}` | `{diff.metric}` | "
            f"{diff.mean_diff_positive_favors_method:+.6f} | {ci} | {wins} | "
            f"{fmt_optional(diff.required_n_for_ci_low_above_zero)} | "
            f"{fmt_optional(diff.required_n_for_eval_sei)} |"
        )

    lines.extend(
        [
            "",
            "## Correlations",
            "",
            "| Horizon | Method | X | Y | Pearson r |",
            "|---:|---|---|---|---:|",
        ]
    )
    for corr in sorted(
        corr_rows,
        key=lambda item: (item.horizon_steps, item.method, item.x_metric),
    ):
        lines.append(
            f"| {corr.horizon_steps} | `{corr.method}` | `{corr.x_metric}` | "
            f"`{corr.y_metric}` | {fmt_optional(corr.pearson_r)} |"
        )

    lines.extend(
        [
            "",
            "## Mechanism Flags",
            "",
            "| Horizon | Method | Gate | Advantage | Active P | "
            "Open gate with negative advantage | Eval improves but final does not | "
            "Eval fast diff |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mechanism in sorted(mechanism_rows, key=lambda item: (item.horizon_steps, item.method)):
        lines.append(
            f"| {mechanism.horizon_steps} | `{mechanism.method}` | "
            f"{fmt_optional(mechanism.final_window_gate_mean)} | "
            f"{fmt_optional(mechanism.final_window_advantage_mean)} | "
            f"{fmt_optional(mechanism.final_window_active_prototypes_mean)} | "
            f"{mechanism.open_gate_negative_advantage} | "
            f"{mechanism.eval_improves_but_final_window_does_not} | "
            f"{fmt_optional(mechanism.eval_fast_diff)} |"
        )

    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "- The post-FFN primary candidate has a real 10000-step held-out signal, "
            "but the lower CI is below the preregistered `0.005` SEI and the "
            "final-window NLL CI still crosses zero.",
            "- The 3000/5000-step post-FFN runs expose the old mechanism failure: "
            "the gate is open while final-window advantage is negative.",
            "- Pre-FFN KV strengthens 10000-step held-out NLL but damages 3000-step "
            "and 10000-step final-window NLL. It is not a clean replacement.",
            "- Fast-only eval does not explain the 10000-step held-out gain; the "
            "deployed memory logits carry most of that signal.",
            "- More seeds alone would not fix the final-window conflict. The next "
            "mechanism should combine the 10000-step held-out retention benefit with "
            "a stricter online-loss safety gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", dest="results")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "outputs/step2_new_directions/validation_variance_diagnosis"
    ))
    return parser.parse_args()


def main() -> None:
    """Run diagnosis."""
    args = parse_args()
    result_paths = tuple(args.results or sorted(args.result_root.glob(DEFAULT_RESULT_GLOB)))
    if not result_paths:
        raise FileNotFoundError("no validation result artifacts found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diffs: list[DiffRow] = []
    correlations: list[CorrelationRow] = []
    mechanisms: list[MechanismRow] = []
    for path in result_paths:
        payload = load_json(path)
        for method in METHODS:
            for metric in (*PRIMARY_METRICS, "eval_fast_nll"):
                diffs.append(diff_row(path, payload, method, metric))
            correlations.extend(correlation_rows(path, payload, method))
            mechanisms.append(mechanism_row(path, payload, method))

    write_csv(args.output_dir / "paired_diffs.csv", diffs)
    write_csv(args.output_dir / "correlations.csv", correlations)
    write_csv(args.output_dir / "mechanism_flags.csv", mechanisms)
    write_markdown(
        args.output_dir / "SUMMARY.md",
        diff_rows=diffs,
        corr_rows=correlations,
        mechanism_rows=mechanisms,
    )
    payload = {
        "results": [str(path) for path in result_paths],
        "paired_diffs": [asdict(row) for row in diffs],
        "correlations": [asdict(row) for row in correlations],
        "mechanism_flags": [asdict(row) for row in mechanisms],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'SUMMARY.md'}")
    print(f"wrote {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()

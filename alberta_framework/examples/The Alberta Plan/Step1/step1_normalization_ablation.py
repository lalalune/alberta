#!/usr/bin/env python3
"""Step 1 normalization ablation.

Tests the effect of online feature normalization (none vs EMA vs Welford vs
BatchNorm-style running statistics) across four optimizers and two streams that
exhibit input-scale variation.

The Alberta Plan paper notes that "the effect of online normalization has
yet to be definitively established in the literature." This script
provides a paired-by-seed comparison to fill that gap.

Hypothesis
----------
Normalization matters MORE on streams with input-scale variation, and
matters LESS for adaptive optimizers (Adam) than for fixed-lr LMS.

Design
------
* Optimizers: LMS(0.01), IDBD(0.05, 0.01), Autostep(0.05, 0.01), Adam(0.001)
* Normalizers: None, EMANormalizer(decay=0.99), WelfordNormalizer(),
  StreamingBatchNormalizer(momentum=0.99)
* Streams:
  - XDistShiftStream (target fixed, scales redrawn every 2000 steps)
  - DynamicScaleShiftStream (both scales and weights change abruptly)
* 30 seeds, 20000 steps per run

Outputs
-------
* outputs/step1_canonical/normalization_ablation_results.json
* outputs/step1_canonical/normalization_ablation_SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.random as jr
import numpy as np

from alberta_framework import (
    IDBD,
    LMS,
    Adam,
    Autostep,
    EMANormalizer,
    LinearLearner,
    StreamingBatchNormalizer,
    Timer,
    WelfordNormalizer,
    run_learning_loop_batched,
)
from alberta_framework.streams.alberta_plan_step1 import XDistShiftStream
from alberta_framework.streams.synthetic import DynamicScaleShiftStream

DEFAULT_NUM_SEEDS = 30
DEFAULT_NUM_STEPS = 20000
WINDOW_LAST = 5000  # Average MSE over the last 5000 steps per seed.


def _make_optimizer(name: str) -> Callable[[], Any]:
    if name == "LMS":
        return lambda: LMS(step_size=0.01)
    if name == "IDBD":
        return lambda: IDBD(initial_step_size=0.05, meta_step_size=0.01)
    if name == "Autostep":
        return lambda: Autostep(
            initial_step_size=0.05, meta_step_size=0.01, tau=10000.0
        )
    if name == "Adam":
        return lambda: Adam(step_size=0.001)
    raise ValueError(f"Unknown optimizer name: {name}")


def _make_normalizer(name: str) -> Callable[[], Any] | None:
    if name == "None":
        return None
    if name == "EMA":
        return lambda: EMANormalizer(decay=0.99)
    if name == "Welford":
        return lambda: WelfordNormalizer()
    if name == "StreamingBatch":
        return lambda: StreamingBatchNormalizer(momentum=0.99)
    raise ValueError(f"Unknown normalizer name: {name}")


def _make_stream(name: str) -> Callable[[], Any]:
    if name == "XDistShift":
        return lambda: XDistShiftStream(
            feature_dim=20,
            num_relevant=5,
            scale_change_interval=2000,
            scale_min=0.1,
            scale_max=10.0,
            noise_std=0.5,
        )
    if name == "DynamicScaleShift":
        return lambda: DynamicScaleShiftStream(
            feature_dim=20,
            scale_change_interval=2000,
            weight_change_interval=1500,
            min_scale=0.01,
            max_scale=100.0,
            noise_std=0.1,
        )
    raise ValueError(f"Unknown stream name: {name}")


def _per_seed_final_window_mse(metrics: jax.Array, window: int) -> np.ndarray:
    """Mean squared error over the last ``window`` steps per seed.

    Args:
        metrics: Array of shape (num_seeds, num_steps, num_cols). Column 0 is
            squared error.
        window: Number of trailing steps to average over.

    Returns:
        np.ndarray of shape (num_seeds,) with NaN replaced by inf so failing
        runs are not silently treated as the best.
    """
    assert metrics.ndim == 3, f"expected (seeds, steps, cols), got {metrics.shape}"
    sq_err = np.asarray(metrics[:, -window:, 0])
    # Treat NaNs (numerical blow-ups) as +inf so they are recognizable as failures
    # without breaking sign/t-tests; we also report the NaN counts in the JSON.
    sq_err = np.where(np.isfinite(sq_err), sq_err, np.inf)
    per_seed = sq_err.mean(axis=1)
    return per_seed


def _paired_t_test(diffs: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t-statistic and p-value.

    Pure-Python implementation (no scipy) using the t-distribution survival
    function via a reasonable normal approximation when n is large.

    Args:
        diffs: Per-seed paired differences.

    Returns:
        (t_stat, p_value). Returns (nan, nan) if the differences are not
        finite or have zero variance.
    """
    finite = diffs[np.isfinite(diffs)]
    n = finite.size
    if n < 2:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    sd = float(finite.std(ddof=1))
    if sd == 0.0:
        return float("nan"), float("nan")
    t = mean / (sd / math.sqrt(n))
    # Two-sided normal-approximation p-value (n=30 is large enough that the
    # t-to-normal approximation is fine for our purposes).
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, p


def _paired_sign_test(diffs: np.ndarray) -> float:
    """Two-sided paired sign-test p-value (normal approximation).

    Counts sign of each finite difference and tests against the null that
    P(positive) = 0.5. Returns NaN if too few finite values.
    """
    finite = diffs[np.isfinite(diffs)]
    n = finite.size
    if n < 2:
        return float("nan")
    pos = int((finite > 0).sum())
    neg = int((finite < 0).sum())
    eff = pos + neg
    if eff < 2:
        return float("nan")
    expected = eff / 2.0
    se = math.sqrt(eff * 0.25)
    z = (max(pos, neg) - 0.5 - expected) / se
    p = math.erfc(z / math.sqrt(2.0))
    return p


def _cohens_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d for paired differences (mean diff / sd diff)."""
    finite = diffs[np.isfinite(diffs)]
    if finite.size < 2:
        return float("nan")
    sd = float(finite.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(finite.mean()) / sd


def _ci95(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    sd = float(finite.std(ddof=1))
    half = 1.96 * sd / math.sqrt(finite.size)
    return mean - half, mean + half


def run(num_seeds: int, num_steps: int, output_dir: Path) -> dict[str, Any]:
    """Run the normalization ablation grid and dump results to ``output_dir``.

    Args:
        num_seeds: Number of independent seeds per cell.
        num_steps: Number of learning steps per seed.
        output_dir: Directory to write JSON and Markdown reports into.

    Returns:
        Aggregated results dictionary (also written to disk).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizers = ["LMS", "IDBD", "Autostep", "Adam"]
    normalizers = ["None", "EMA", "Welford", "StreamingBatch"]
    streams = ["XDistShift", "DynamicScaleShift"]
    window = min(WINDOW_LAST, num_steps)

    keys = jr.split(jr.key(0xA17), num_seeds)

    per_run: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, Any]] = {}

    total_cells = len(optimizers) * len(normalizers) * len(streams)
    cell_idx = 0
    with Timer("normalization_ablation", verbose=True) as outer_timer:
        for stream_name in streams:
            stream_factory = _make_stream(stream_name)
            for opt_name in optimizers:
                opt_factory = _make_optimizer(opt_name)

                # First collect per-(normalizer) per-seed final-window MSE
                per_norm_arrays: dict[str, np.ndarray] = {}

                for norm_name in normalizers:
                    cell_idx += 1
                    cell_key = f"{stream_name}|{opt_name}|{norm_name}"
                    norm_factory = _make_normalizer(norm_name)

                    learner = LinearLearner(
                        optimizer=opt_factory(),
                        normalizer=None if norm_factory is None else norm_factory(),
                    )
                    stream = stream_factory()

                    print(
                        f"[{cell_idx:>2}/{total_cells}] {cell_key} "
                        f"(seeds={num_seeds}, steps={num_steps})",
                        flush=True,
                    )
                    with Timer(f"  -> {cell_key}", verbose=True):
                        result = run_learning_loop_batched(
                            learner=learner,
                            stream=stream,
                            num_steps=num_steps,
                            keys=keys,
                        )
                        # Block until done so the timer is meaningful.
                        result.metrics.block_until_ready()

                    per_seed = _per_seed_final_window_mse(result.metrics, window)
                    per_norm_arrays[norm_name] = per_seed

                    finite_mask = np.isfinite(per_seed)
                    finite_vals = per_seed[finite_mask]
                    nan_count = int((~finite_mask).sum())
                    mean = (
                        float(finite_vals.mean()) if finite_vals.size > 0 else float("nan")
                    )
                    std = (
                        float(finite_vals.std(ddof=1))
                        if finite_vals.size > 1
                        else float("nan")
                    )
                    median = (
                        float(np.median(finite_vals))
                        if finite_vals.size > 0
                        else float("nan")
                    )
                    ci_lo, ci_hi = _ci95(per_seed)
                    per_run[cell_key] = {
                        "stream": stream_name,
                        "optimizer": opt_name,
                        "normalizer": norm_name,
                        "num_seeds": int(per_seed.size),
                        "nan_seeds": nan_count,
                        "mean_final_window_mse": mean,
                        "median_final_window_mse": median,
                        "std_final_window_mse": std,
                        "ci95_lo": ci_lo,
                        "ci95_hi": ci_hi,
                        "per_seed_final_window_mse": [
                            (float(v) if math.isfinite(v) else None) for v in per_seed
                        ],
                    }

                # Paired-by-seed differences for each normalizer pair.
                # `inf - inf` legitimately produces NaN (both runs blew up); we
                # filter to finite pairs in every consumer below, so silence
                # the harmless RuntimeWarning here.
                normalizer_pairs = [
                    ("None", "EMA"),
                    ("None", "Welford"),
                    ("None", "StreamingBatch"),
                    ("EMA", "Welford"),
                    ("EMA", "StreamingBatch"),
                    ("Welford", "StreamingBatch"),
                ]
                for a, b in normalizer_pairs:
                    with np.errstate(invalid="ignore"):
                        diff = per_norm_arrays[a] - per_norm_arrays[b]
                    # Positive => `a` (e.g., "None") MSE higher than `b` (e.g., "EMA").
                    t_stat, t_p = _paired_t_test(diff)
                    sign_p = _paired_sign_test(diff)
                    d = _cohens_d_paired(diff)
                    finite = diff[np.isfinite(diff)]
                    mean_diff = float(finite.mean()) if finite.size else float("nan")
                    se_diff = (
                        float(finite.std(ddof=1) / math.sqrt(finite.size))
                        if finite.size > 1
                        else float("nan")
                    )

                    a_finite = per_norm_arrays[a][np.isfinite(per_norm_arrays[a])]
                    b_finite = per_norm_arrays[b][np.isfinite(per_norm_arrays[b])]
                    if a_finite.size and b_finite.size and float(b_finite.mean()) != 0.0:
                        # Percent improvement of `b` over `a`: positive means b lower MSE.
                        pct_improvement = 100.0 * (
                            float(a_finite.mean()) - float(b_finite.mean())
                        ) / float(a_finite.mean())
                    else:
                        pct_improvement = float("nan")

                    paired_key = f"{stream_name}|{opt_name}|{a}_vs_{b}"
                    paired[paired_key] = {
                        "stream": stream_name,
                        "optimizer": opt_name,
                        "left": a,
                        "right": b,
                        "num_paired_seeds": int(diff.size),
                        "num_finite_pairs": int(finite.size),
                        "mean_diff_left_minus_right": mean_diff,
                        "se_diff": se_diff,
                        "cohens_d_paired": d,
                        "paired_t_stat": t_stat,
                        "paired_t_p_value": t_p,
                        "paired_sign_p_value": sign_p,
                        "pct_improvement_of_right_over_left": pct_improvement,
                    }

    aggregated: dict[str, Any] = {
        "config": {
            "num_seeds": num_seeds,
            "num_steps": num_steps,
            "window_last": window,
            "optimizers": {
                "LMS": {"type": "LMS", "step_size": 0.01},
                "IDBD": {
                    "type": "IDBD",
                    "initial_step_size": 0.05,
                    "meta_step_size": 0.01,
                },
                "Autostep": {
                    "type": "Autostep",
                    "initial_step_size": 0.05,
                    "meta_step_size": 0.01,
                    "tau": 10000.0,
                },
                "Adam": {"type": "Adam", "step_size": 0.001},
            },
            "normalizers": {
                "None": None,
                "EMA": {"type": "EMANormalizer", "decay": 0.99},
                "Welford": {"type": "WelfordNormalizer"},
                "StreamingBatch": {
                    "type": "StreamingBatchNormalizer",
                    "momentum": 0.99,
                },
            },
            "streams": {
                "XDistShift": {
                    "type": "XDistShiftStream",
                    "feature_dim": 20,
                    "num_relevant": 5,
                    "scale_change_interval": 2000,
                    "scale_min": 0.1,
                    "scale_max": 10.0,
                    "noise_std": 0.5,
                },
                "DynamicScaleShift": {
                    "type": "DynamicScaleShiftStream",
                    "feature_dim": 20,
                    "scale_change_interval": 2000,
                    "weight_change_interval": 1500,
                    "min_scale": 0.01,
                    "max_scale": 100.0,
                    "noise_std": 0.1,
                },
            },
            "num_seeds_actual": num_seeds,
            "wall_clock_seconds": outer_timer.duration,
        },
        "per_run": per_run,
        "paired": paired,
    }

    json_path = output_dir / "normalization_ablation_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(aggregated, fh, indent=2)
    print(f"\nWrote {json_path}")

    md_path = output_dir / "normalization_ablation_SUMMARY.md"
    md_path.write_text(_render_summary_markdown(aggregated), encoding="utf-8")
    print(f"Wrote {md_path}")

    return aggregated


def _render_summary_markdown(aggregated: dict[str, Any]) -> str:
    """Render the human-readable Markdown summary."""
    cfg = aggregated["config"]
    lines: list[str] = []
    lines.append("# Step 1 Normalization Ablation Summary")
    lines.append("")
    lines.append(
        f"- Seeds: {cfg['num_seeds']}, steps per run: {cfg['num_steps']},"
        f" final-window: last {cfg['window_last']} steps."
    )
    lines.append(
        f"- Wall-clock: {cfg['wall_clock_seconds']:.1f} s."
    )
    lines.append("")
    lines.append("## EMA vs no normalization (% MSE improvement, paired by seed)")
    lines.append("")
    lines.append(
        "Cells show `pct_improvement (95% CI on per-seed difference)`."
        " Positive = EMANormalizer beats None."
    )
    lines.append("")

    optimizers = ["LMS", "IDBD", "Autostep", "Adam"]
    streams = list(cfg["streams"].keys())

    def _format_cell(
        left_run: dict[str, Any],
        right_run: dict[str, Any],
        paired_entry: dict[str, Any],
    ) -> str:
        left_arr = np.asarray(
            [v if v is not None else np.nan for v in left_run["per_seed_final_window_mse"]]
        )
        right_arr = np.asarray(
            [v if v is not None else np.nan for v in right_run["per_seed_final_window_mse"]]
        )
        with np.errstate(invalid="ignore"):
            diff = left_arr - right_arr
        finite_mask = np.isfinite(diff)
        n_finite = int(finite_mask.sum())
        n_total = int(diff.size)
        if n_finite < 2:
            left_unstable = int(left_run["nan_seeds"])
            right_unstable = int(right_run["nan_seeds"])
            return (
                f"unstable (left {left_unstable}/{n_total} NaN, "
                f"right {right_unstable}/{n_total} NaN)"
            )
        ci_lo, ci_hi = _ci95(diff)
        pct = paired_entry["pct_improvement_of_right_over_left"]
        d = paired_entry["cohens_d_paired"]
        p = paired_entry["paired_t_p_value"]
        nfinite_note = "" if n_finite == n_total else f", n_finite={n_finite}/{n_total}"
        # Cap ridiculous percentages caused by tiny baseline means (the raw
        # value lives in the JSON; the cap is a display convenience).
        if math.isfinite(pct) and abs(pct) >= 1000.0:
            pct_str = ">+1000%" if pct > 0 else "<-1000%"
        else:
            pct_str = f"{pct:+.1f}%"
        return (
            f"{pct_str} (Δ CI [{ci_lo:+.3g}, {ci_hi:+.3g}],"
            f" d={d:+.2f}, p={p:.2g}{nfinite_note})"
        )

    header_cells = ["Optimizer"] + streams
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cells) + " |")
    for opt in optimizers:
        row = [opt]
        for stream in streams:
            none_run = aggregated["per_run"][f"{stream}|{opt}|None"]
            ema_run = aggregated["per_run"][f"{stream}|{opt}|EMA"]
            paired_entry = aggregated["paired"][f"{stream}|{opt}|None_vs_EMA"]
            row.append(_format_cell(none_run, ema_run, paired_entry))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Welford vs no normalization (% MSE improvement, paired by seed)")
    lines.append("")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cells) + " |")
    for opt in optimizers:
        row = [opt]
        for stream in streams:
            none_run = aggregated["per_run"][f"{stream}|{opt}|None"]
            wel_run = aggregated["per_run"][f"{stream}|{opt}|Welford"]
            paired_entry = aggregated["paired"][f"{stream}|{opt}|None_vs_Welford"]
            row.append(_format_cell(none_run, wel_run, paired_entry))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## StreamingBatch vs no normalization (% MSE improvement, paired by seed)")
    lines.append("")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cells) + " |")
    for opt in optimizers:
        row = [opt]
        for stream in streams:
            none_run = aggregated["per_run"][f"{stream}|{opt}|None"]
            batch_run = aggregated["per_run"][f"{stream}|{opt}|StreamingBatch"]
            paired_entry = aggregated["paired"][f"{stream}|{opt}|None_vs_StreamingBatch"]
            row.append(_format_cell(none_run, batch_run, paired_entry))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    def _fmt_mean(run: dict[str, Any]) -> str:
        v = run["mean_final_window_mse"]
        unstable = int(run["nan_seeds"])
        total = int(run["num_seeds"])
        if not math.isfinite(v):
            return f"unstable ({unstable}/{total} NaN)"
        if unstable == 0:
            return f"{v:.4g}"
        return f"{v:.4g} ({unstable}/{total} NaN)"

    lines.append("## Per-cell mean (final-window MSE)")
    lines.append("")
    lines.append("| Stream | Optimizer | None | EMA | Welford | StreamingBatch |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for stream in streams:
        for opt in optimizers:
            none_run = aggregated["per_run"][f"{stream}|{opt}|None"]
            ema_run = aggregated["per_run"][f"{stream}|{opt}|EMA"]
            wel_run = aggregated["per_run"][f"{stream}|{opt}|Welford"]
            batch_run = aggregated["per_run"][f"{stream}|{opt}|StreamingBatch"]
            lines.append(
                f"| {stream} | {opt} | {_fmt_mean(none_run)} | "
                f"{_fmt_mean(ema_run)} | {_fmt_mean(wel_run)} | "
                f"{_fmt_mean(batch_run)} |"
            )
    lines.append("")

    # Headline findings panel
    lines.append("## Headline findings")
    lines.append("")
    headlines: list[str] = []
    for stream in streams:
        for opt in optimizers:
            none_run = aggregated["per_run"][f"{stream}|{opt}|None"]
            ema_run = aggregated["per_run"][f"{stream}|{opt}|EMA"]
            wel_run = aggregated["per_run"][f"{stream}|{opt}|Welford"]
            batch_run = aggregated["per_run"][f"{stream}|{opt}|StreamingBatch"]
            none_unstable = none_run["nan_seeds"] == none_run["num_seeds"]
            ema_unstable = ema_run["nan_seeds"] == ema_run["num_seeds"]
            wel_unstable = wel_run["nan_seeds"] == wel_run["num_seeds"]
            batch_unstable = batch_run["nan_seeds"] == batch_run["num_seeds"]
            if none_unstable and not (ema_unstable and wel_unstable and batch_unstable):
                stable_norm = "EMA"
                stable_run = ema_run
                for stable_norm, stable_run, unstable in [
                    ("EMA", ema_run, ema_unstable),
                    ("Welford", wel_run, wel_unstable),
                    ("StreamingBatch", batch_run, batch_unstable),
                ]:
                    if not unstable:
                        break
                stable_mean = stable_run["mean_final_window_mse"]
                headlines.append(
                    f"- `{opt}` on `{stream}`: 100% of seeds NaN without "
                    f"normalization; with `{stable_norm}` MSE = {stable_mean:.4g}"
                    f" (normalization is REQUIRED, not just helpful)."
                )
    if not headlines:
        headlines.append("- No optimizer was destabilised solely by lack of normalization.")
    lines.extend(headlines)
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `pct_improvement_of_right_over_left` is computed from per-cell"
        " means, while the CI/Cohen's d/p-value characterise the per-seed"
        " paired distribution `MSE(no-norm) - MSE(normalizer)`."
    )
    lines.append(
        "- Paired t-tests use the normal approximation. With 30 seeds the"
        " approximation is good but we also report the sign-test p-value"
        " in the JSON (`paired_sign_p_value`)."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "outputs" / "step1_canonical",
    )
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    args = parser.parse_args()

    run(
        num_seeds=args.num_seeds,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

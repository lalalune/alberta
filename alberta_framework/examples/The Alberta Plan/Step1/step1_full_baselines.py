#!/usr/bin/env python3
"""Multi-baseline replication of Alberta Plan Step 1.

Runs every optimizer named in Alberta Plan footnote 11 (Sutton, Bowling &
Pilarski 2022) — LMS, IDBD, Autostep, Adam, RMSprop, NADALINE — plus the
public Degris-coauthored AdaGain method (Jacobsen et al. 2019) against the
four canonical Step 1 testbeds with at least 30 seeds, then computes
paired-difference statistics versus best-tuned LMS. Tuning is over the joint
optimizer-hyperparameter and online-normalizer grid, so the canonical claim is
not biased against methods that require feature normalization.

Streams covered:

1. ``Sutton1992_noiseless`` — original Sutton 1992 Experiment 1
   (``noise_std=0``).
2. ``Sutton1992_noisy`` — Alberta Plan version with the η_t noise term
   (``noise_std=1.0``).
3. ``AlbertaPlanStep1`` — full canonical Step 1 task: drifting ``w*_t``,
   drifting ``b*_t``, additive noise.
4. ``XDistShift`` — input-distribution non-stationarity (per-feature scales
   redrawn every 2,000 steps).

For each (stream, optimizer, hyperparameter, seed) the script measures the
mean squared error over the last ``measurement_steps`` of a
``burn_in_steps + measurement_steps`` run. The best hyperparameter for each
(stream, optimizer) pair is chosen by lowest mean MSE across seeds. Paired
differences (LMS_best − other_best, per seed) yield mean ± stderr,
wins-out-of-N, and Cohen's d.

Outputs (relative to project root):

* ``outputs/step1_canonical/multi_baseline_results.json`` — full machine-
  readable results (per-run + tuned + paired stats).
* ``outputs/step1_canonical/SUMMARY.md`` — human-readable summary tables and
  the headline statement.

Run from the project root with the venv active::

    python "examples/The Alberta Plan/Step1/step1_full_baselines.py"

Optional flags::

    --output-dir DIR     Override the output directory.
    --seeds N            Override the number of seeds (default 30).
    --burn-in N          Burn-in steps (default 20000).
    --measurement N      Measurement steps (default 10000).
    --normalizers LIST   Comma-separated normalizers (default all).
    --streams LIST       Comma-separated streams (default all).
    --optimizers LIST    Comma-separated optimizers (default all).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    IDBD,
    LMS,
    NADALINE,
    AdaGain,
    Adam,
    AlbertaPlanStep1Stream,
    Autostep,
    AutostepGTDLambda,
    EMANormalizer,
    LinearLearner,
    RMSprop,
    StreamingBatchNormalizer,
    SuttonExperiment1Stream,
    Timer,
    WelfordNormalizer,
    XDistShiftStream,
)

# ---------------------------------------------------------------------------
# Optimizer / stream specifications
# ---------------------------------------------------------------------------


OPTIMIZER_GRIDS: dict[str, list[dict[str, float]]] = {
    "LMS": [
        {"step_size": a}
        for a in [0.0003, 0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1]
    ],
    "IDBD": [
        {"initial_step_size": a, "meta_step_size": t}
        for a in [0.005, 0.02, 0.05]
        for t in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    ],
    "Autostep": [
        {"initial_step_size": a, "meta_step_size": m, "tau": 10000.0}
        for a in [0.005, 0.02, 0.05]
        for m in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    ],
    # Autostep-for-GTD(lambda) (Kearney et al. 2019), supervised limit.
    # Closes Alberta Plan footnote 11 by name. In supervised mode this is
    # numerically equivalent to Autostep; we sweep the same grid so the
    # comparison is symmetric.
    "AutostepGTDLambda": [
        {"initial_step_size": a, "meta_step_size": m, "tau": 10000.0}
        for a in [0.005, 0.02, 0.05]
        for m in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    ],
    "AdaGain": [
        {"initial_step_size": 0.05, "meta_step_size": m, "forgetting_rate": 0.1}
        for m in [0.0001, 0.001, 0.01]
    ]
    + [
        {"initial_step_size": 0.1, "meta_step_size": m, "forgetting_rate": 0.1}
        for m in [0.0001, 0.001, 0.01]
    ],
    "Adam": [
        {"step_size": lr}
        for lr in [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]
    ],
    "RMSprop": [
        {"step_size": lr}
        for lr in [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]
    ],
    "NADALINE": [
        {"step_size": lr}
        for lr in [0.0003, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
    ],
}

NORMALIZER_GRIDS: dict[str, dict[str, float] | None] = {
    "None": None,
    "EMA": {"decay": 0.99},
    "Welford": {},
    "StreamingBatch": {"momentum": 0.99},
}


def build_optimizer(name: str, hp: dict[str, float]):
    """Return a fresh optimizer instance for the given name/hyperparams."""
    if name == "LMS":
        return LMS(**hp)
    if name == "IDBD":
        return IDBD(**hp)
    if name == "Autostep":
        return Autostep(**hp)
    if name == "AutostepGTDLambda":
        return AutostepGTDLambda(**hp)
    if name == "AdaGain":
        return AdaGain(**hp)
    if name == "Adam":
        return Adam(**hp)
    if name == "RMSprop":
        return RMSprop(**hp)
    if name == "NADALINE":
        return NADALINE(**hp)
    raise ValueError(f"Unknown optimizer: {name}")


def build_normalizer(name: str):
    """Return a fresh normalizer instance, or None for raw features."""
    if name == "None":
        return None
    hp = NORMALIZER_GRIDS[name]
    if name == "EMA":
        return EMANormalizer(**(hp or {}))
    if name == "Welford":
        return WelfordNormalizer(**(hp or {}))
    if name == "StreamingBatch":
        return StreamingBatchNormalizer(**(hp or {}))
    raise ValueError(f"Unknown normalizer: {name}")


def build_streams() -> dict[str, Any]:
    """Return the four canonical Alberta Plan Step 1 testbeds."""
    return {
        "Sutton1992_noiseless": SuttonExperiment1Stream(
            num_relevant=5,
            num_irrelevant=15,
            change_interval=20,
            noise_std=0.0,
        ),
        "Sutton1992_noisy": SuttonExperiment1Stream(
            num_relevant=5,
            num_irrelevant=15,
            change_interval=20,
            noise_std=1.0,
        ),
        "AlbertaPlanStep1": AlbertaPlanStep1Stream(
            feature_dim=20,
            num_relevant=5,
            drift_rate_w=0.001,
            drift_rate_b=0.001,
            noise_std=1.0,
        ),
        "XDistShift": XDistShiftStream(
            feature_dim=20,
            num_relevant=5,
            scale_change_interval=2000,
            scale_min=0.1,
            scale_max=10.0,
            noise_in_target=True,
        ),
    }


# ---------------------------------------------------------------------------
# Core experiment runner (vmapped over seeds)
# ---------------------------------------------------------------------------


def _single_seed_mean_mse(
    learner: LinearLearner,
    stream: Any,
    burn_in_steps: int,
    measurement_steps: int,
    key: Any,
) -> Any:
    """Run a single seed end-to-end and return mean squared error on the tail.

    Uses one ``jax.lax.scan`` over the full ``burn_in + measurement`` horizon
    so JIT compilation happens once per (optimizer, stream) cell rather than
    twice (once for burn-in, once for measurement) — and so we can plumb
    the post-burn-in learner state through without dealing with a
    pre-vmapped initial state.

    Args:
        learner: Fresh ``LinearLearner`` instance.
        stream: ``ScanStream`` instance.
        burn_in_steps: Steps to discard before measuring.
        measurement_steps: Steps to average MSE over.
        key: JAX random key.

    Returns:
        Scalar JAX array — mean squared error over the measurement window.
    """
    learner_state = learner.init(stream.feature_dim)
    stream_state = stream.init(key)
    total_steps = burn_in_steps + measurement_steps

    def step_fn(carry, idx):
        l_state, s_state = carry
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)
        # metrics[0] is squared_error.
        return (result.state, new_s_state), result.metrics[0]

    (_, _), squared_errors = jax.lax.scan(
        step_fn, (learner_state, stream_state), jnp.arange(total_steps)
    )
    tail = squared_errors[burn_in_steps:]
    return jnp.mean(tail)


def run_one_config(
    stream: Any,
    optimizer_name: str,
    hp: dict[str, float],
    normalizer_name: str,
    burn_in_steps: int,
    measurement_steps: int,
    seeds: list[int],
) -> np.ndarray:
    """Run a single (stream, optimizer, hp, normalizer) cell across all seeds.

    Vmaps a single-seed end-to-end run over the seed dimension. Returns
    per-seed mean MSE over the tail (measurement window).

    Args:
        stream: A ``ScanStream`` instance.
        optimizer_name: Name of the optimizer (key into ``OPTIMIZER_GRIDS``).
        hp: Hyperparameter dict.
        normalizer_name: Name of the normalizer (key into ``NORMALIZER_GRIDS``).
        burn_in_steps: Steps to discard.
        measurement_steps: Steps to average MSE over.
        seeds: List of integer seeds.

    Returns:
        ``numpy.ndarray`` of shape ``(len(seeds),)`` with the mean MSE per
        seed over the measurement window. Non-finite entries (numerical
        blow-up) are replaced with ``+inf`` so they never win tuning.
    """
    learner = LinearLearner(
        optimizer=build_optimizer(optimizer_name, hp),
        normalizer=build_normalizer(normalizer_name),
    )

    def per_seed(key):
        return _single_seed_mean_mse(
            learner, stream, burn_in_steps, measurement_steps, key,
        )

    keys = jnp.stack([jr.key(int(s)) for s in seeds])
    mse_per_seed = jax.vmap(per_seed)(keys)
    mse_per_seed.block_until_ready()
    mse_np = np.asarray(mse_per_seed)
    mse_np = np.where(np.isfinite(mse_np), mse_np, np.inf)
    return mse_np


# ---------------------------------------------------------------------------
# Aggregation and statistics
# ---------------------------------------------------------------------------


def hp_label(hp: dict[str, float]) -> str:
    """Human-readable label for a hyperparameter dict."""
    return ", ".join(f"{k}={v:g}" for k, v in hp.items())


def aggregate_per_run(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best hyperparameter per (stream, optimizer)."""
    grouped: dict[tuple[str, str, str, str], list[tuple[int, float]]] = {}
    hp_lookup: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in per_run:
        key = (
            row["stream"],
            row["optimizer"],
            row["normalizer"],
            hp_label(row["hp"]),
        )
        grouped.setdefault(key, []).append((row["seed"], row["final_mse"]))
        hp_lookup[key] = row["hp"]

    # Per-cell mean MSE and per-seed array.
    cell_stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, entries in grouped.items():
        entries.sort(key=lambda e: e[0])
        seeds = [e[0] for e in entries]
        mses = np.array([e[1] for e in entries], dtype=np.float64)
        finite = mses[np.isfinite(mses)]
        mean = float(np.mean(finite)) if len(finite) > 0 else float("inf")
        stderr = (
            float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
            if len(finite) > 1
            else 0.0
        )
        cell_stats[key] = {
            "hp": hp_lookup[key],
            "mean_mse": mean,
            "stderr": stderr,
            "per_seed": [float(m) for m in mses.tolist()],
            "seeds": seeds,
        }

    # Pick the best HP per (stream, optimizer).
    tuned: dict[str, dict[str, dict[str, Any]]] = {}
    for (stream, optimizer, normalizer, _hp_lab), stats in cell_stats.items():
        bucket = tuned.setdefault(stream, {})
        existing = bucket.get(optimizer)
        if existing is None or stats["mean_mse"] < existing["mean_mse"]:
            stats = dict(stats)
            stats["normalizer"] = normalizer
            bucket[optimizer] = stats

    return {"tuned": tuned, "all_cells": cell_stats}


def cohens_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d on paired differences (mean / std of the difference)."""
    finite = diffs[np.isfinite(diffs)]
    if len(finite) < 2:
        return 0.0
    sd = float(np.std(finite, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(finite) / sd)


def paired_vs_lms(tuned: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Compute paired-difference statistics versus best-tuned LMS."""
    paired: dict[str, dict[str, Any]] = {}
    for stream, opt_results in tuned.items():
        if "LMS" not in opt_results:
            continue
        lms = np.asarray(opt_results["LMS"]["per_seed"], dtype=np.float64)
        per_optim: dict[str, Any] = {}
        for opt, stats in opt_results.items():
            if opt == "LMS":
                continue
            other = np.asarray(stats["per_seed"], dtype=np.float64)
            n = min(len(lms), len(other))
            diffs = lms[:n] - other[:n]
            finite_mask = np.isfinite(diffs)
            n_finite = int(np.sum(finite_mask))
            if n_finite == 0:
                per_optim[opt] = {
                    "mean_diff": float("nan"),
                    "stderr_diff": float("nan"),
                    "wins": 0,
                    "n_seeds": 0,
                    "cohens_d": 0.0,
                }
                continue
            d_finite = diffs[finite_mask]
            mean_diff = float(np.mean(d_finite))
            stderr = (
                float(np.std(d_finite, ddof=1) / np.sqrt(len(d_finite)))
                if len(d_finite) > 1
                else 0.0
            )
            wins = int(np.sum(d_finite > 0))
            per_optim[opt] = {
                "mean_diff": mean_diff,
                "stderr_diff": stderr,
                "wins": wins,
                "n_seeds": n_finite,
                "cohens_d": cohens_d_paired(diffs),
            }
        paired[stream] = per_optim
    return paired


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_float(x: float, prec: int = 4) -> str:
    """Format a float with consistent precision; show inf/NaN explicitly."""
    if not np.isfinite(x):
        return "NaN" if np.isnan(x) else "inf"
    return f"{x:.{prec}f}"


def write_summary_md(
    output_path: Path,
    config: dict[str, Any],
    aggregated: dict[str, Any],
    paired: dict[str, dict[str, Any]],
    wall_clock_s: float,
) -> None:
    """Write the human-readable Markdown summary."""
    tuned = aggregated["tuned"]
    streams = list(tuned.keys())
    optimizers = sorted({opt for s in streams for opt in tuned[s]})

    lines: list[str] = []
    lines.append("# Alberta Plan Step 1 — Multi-Baseline Results")
    lines.append("")
    lines.append(
        "Replication of the Alberta Plan Step 1 supervised-learning task across all "
        "optimizers named in Sutton, Bowling & Pilarski 2022 footnote 11 "
        "(LMS, IDBD, Autostep, Adam, RMSprop, NADALINE), plus public AdaGain "
        "(Jacobsen et al. 2019), evaluated on four non-stationary streams."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Seeds: {config['seeds']}")
    lines.append(f"- Burn-in steps: {config['burn_in']:,}")
    lines.append(f"- Measurement steps: {config['measurement']:,}")
    lines.append(f"- Normalizers: {', '.join(config['normalizers'])}")
    lines.append(f"- Total wall-clock: {wall_clock_s / 60.0:.2f} minutes")
    lines.append("")
    lines.append("Optimizer hyperparameter grids:")
    lines.append("")
    for opt, grid in config["optimizer_grids"].items():
        labels = [hp_label(hp) for hp in grid]
        lines.append(f"- **{opt}**: {labels}")
    lines.append("")

    # Best-tuned MSE table per stream.
    lines.append("## Best-tuned MSE per (optimizer, stream)")
    lines.append("")
    header = ["Optimizer"] + streams
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for opt in optimizers:
        row = [opt]
        for stream in streams:
            entry = tuned[stream].get(opt)
            if entry is None:
                row.append("—")
            else:
                row.append(
                    f"{fmt_float(entry['mean_mse'])} ± {fmt_float(entry['stderr'])}"
                )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Best HPs.
    lines.append("## Selected hyperparameters and normalizers (best per cell)")
    lines.append("")
    lines.append("| Optimizer | " + " | ".join(streams) + " |")
    lines.append("|" + "|".join(["---"] * (len(streams) + 1)) + "|")
    for opt in optimizers:
        row = [opt]
        for stream in streams:
            entry = tuned[stream].get(opt)
            row.append(
                f"{hp_label(entry['hp'])}; norm={entry['normalizer']}"
                if entry is not None
                else "—"
            )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Paired-difference table.
    lines.append("## Paired differences vs. best-tuned LMS")
    lines.append("")
    lines.append(
        "Each cell shows ``mean(LMS − other) ± stderr`` over paired seeds, "
        "with wins-out-of-N (positive = the alternative beat LMS on that seed) "
        "and Cohen's d on the paired differences."
    )
    lines.append("")
    for stream in streams:
        lines.append(f"### {stream}")
        lines.append("")
        lines.append("| Optimizer | mean diff | stderr | wins | n | Cohen's d |")
        lines.append("|---|---|---|---|---|---|")
        per_optim = paired.get(stream, {})
        for opt in optimizers:
            if opt == "LMS":
                continue
            stats = per_optim.get(opt)
            if stats is None:
                continue
            lines.append(
                "| {opt} | {md} | {se} | {w}/{n} | {n} | {d} |".format(
                    opt=opt,
                    md=fmt_float(stats["mean_diff"]),
                    se=fmt_float(stats["stderr_diff"]),
                    w=stats["wins"],
                    n=stats["n_seeds"],
                    d=fmt_float(stats["cohens_d"], 3),
                )
            )
        lines.append("")

    # Headline statement.
    lines.append("## Headline")
    lines.append("")
    headlines: list[str] = []
    for stream in streams:
        per_optim = paired.get(stream, {})
        winners: list[tuple[str, dict[str, Any]]] = []
        for opt, stats in per_optim.items():
            if not np.isfinite(stats["mean_diff"]) or stats["n_seeds"] == 0:
                continue
            margin_in_sigmas = (
                stats["mean_diff"] / stats["stderr_diff"]
                if stats["stderr_diff"] > 0
                else float("inf")
            )
            if stats["mean_diff"] > 0 and margin_in_sigmas >= 2.0:
                winners.append((opt, stats))
        if not winners:
            headlines.append(
                f"- **{stream}**: no optimizer beat best-tuned LMS by ≥ 2σ."
            )
            continue
        winners.sort(key=lambda kv: -kv[1]["mean_diff"])
        best_opt, best = winners[0]
        lms_entry = tuned[stream]["LMS"]
        rel_pct = (
            100.0 * best["mean_diff"] / lms_entry["mean_mse"]
            if lms_entry["mean_mse"] > 0
            else float("nan")
        )
        headlines.append(
            "- **{stream}**: {opt} beat best-tuned LMS by "
            "{md} MSE ({rel:.1f}% relative, {w}/{n} seed wins, "
            "Cohen's d = {d}).".format(
                stream=stream,
                opt=best_opt,
                md=fmt_float(best["mean_diff"]),
                rel=rel_pct,
                w=best["wins"],
                n=best["n_seeds"],
                d=fmt_float(best["cohens_d"], 3),
            )
        )
    lines.extend(headlines)
    lines.append("")

    output_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    output_dir: str,
    seeds: int = 30,
    burn_in: int = 20000,
    measurement: int = 10000,
    normalizers: list[str] | None = None,
    stream_names: list[str] | None = None,
    optimizer_names: list[str] | None = None,
) -> None:
    """Run the full multi-baseline sweep."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalizer_names = list(NORMALIZER_GRIDS) if normalizers is None else normalizers
    unknown_normalizers = sorted(set(normalizer_names) - set(NORMALIZER_GRIDS))
    if unknown_normalizers:
        raise ValueError(f"Unknown normalizers: {unknown_normalizers}")

    seed_list = list(range(seeds))
    all_streams = build_streams()
    requested_streams = list(all_streams) if stream_names is None else stream_names
    unknown_streams = sorted(set(requested_streams) - set(all_streams))
    if unknown_streams:
        raise ValueError(f"Unknown streams: {unknown_streams}")
    streams = {name: all_streams[name] for name in requested_streams}
    stream_keys = list(streams.keys())
    requested_optimizers = (
        list(OPTIMIZER_GRIDS) if optimizer_names is None else optimizer_names
    )
    unknown_optimizers = sorted(set(requested_optimizers) - set(OPTIMIZER_GRIDS))
    if unknown_optimizers:
        raise ValueError(f"Unknown optimizers: {unknown_optimizers}")
    optimizer_grid = (
        {name: OPTIMIZER_GRIDS[name] for name in requested_optimizers}
    )

    config = {
        "seeds": seeds,
        "burn_in": burn_in,
        "measurement": measurement,
        "stream_keys": stream_keys,
        "normalizers": normalizer_names,
        "normalizer_grids": NORMALIZER_GRIDS,
        "optimizer_grids": {
            opt: [dict(hp) for hp in grid] for opt, grid in optimizer_grid.items()
        },
    }

    print("=" * 78)
    print("Alberta Plan Step 1 — multi-baseline replication")
    print("=" * 78)
    print(
        f"Seeds: {seeds} | burn-in: {burn_in:,} | measurement: {measurement:,} | "
        f"streams: {len(streams)} | optimizers: {len(optimizer_grid)} | "
        f"normalizers: {len(normalizer_names)}"
        ,
        flush=True,
    )
    print()

    per_run: list[dict[str, Any]] = []
    t_total_start = time.time()

    for stream_name, stream in streams.items():
        with Timer(f"Stream {stream_name}"):
            for opt_name, grid in optimizer_grid.items():
                t_opt_start = time.time()
                for normalizer_name in normalizer_names:
                    for hp in grid:
                        mse_per_seed = run_one_config(
                            stream,
                            opt_name,
                            hp,
                            normalizer_name,
                            burn_in_steps=burn_in,
                            measurement_steps=measurement,
                            seeds=seed_list,
                        )
                        for seed, mse in zip(seed_list, mse_per_seed, strict=True):
                            per_run.append(
                                {
                                    "stream": stream_name,
                                    "optimizer": opt_name,
                                    "normalizer": normalizer_name,
                                    "hp": dict(hp),
                                    "seed": int(seed),
                                    "final_mse": float(mse),
                                }
                            )
                dt = time.time() - t_opt_start
                print(
                    f"  [{stream_name}] {opt_name}: {len(grid)} hp x "
                    f"{len(normalizer_names)} normalizers x {seeds} seeds "
                    f"in {dt:.1f}s"
                )

    aggregated = aggregate_per_run(per_run)
    paired = paired_vs_lms(aggregated["tuned"])
    wall_clock_s = time.time() - t_total_start

    # Write JSON.
    out_json = output_path / "multi_baseline_results.json"
    payload = {
        "config": config,
        "wall_clock_s": wall_clock_s,
        "per_run": per_run,
        "tuned": {
            stream: {
                opt: {
                    "hp": entry["hp"],
                    "normalizer": entry["normalizer"],
                    "mean_mse": entry["mean_mse"],
                    "stderr": entry["stderr"],
                    "per_seed": entry["per_seed"],
                    "seeds": entry["seeds"],
                }
                for opt, entry in opt_results.items()
            }
            for stream, opt_results in aggregated["tuned"].items()
        },
        "paired_vs_lms": paired,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_json}")

    # Write Markdown summary.
    out_md = output_path / "SUMMARY.md"
    write_summary_md(out_md, config, aggregated, paired, wall_clock_s)
    print(f"Wrote {out_md}")
    print(f"\nTotal wall-clock: {wall_clock_s / 60.0:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/step1_canonical",
        help="Directory to write results into.",
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--burn-in", type=int, default=20000)
    parser.add_argument("--measurement", type=int, default=10000)
    parser.add_argument(
        "--normalizers",
        type=str,
        default=",".join(NORMALIZER_GRIDS),
        help=(
            "Comma-separated normalizer names. Valid values: "
            + ", ".join(NORMALIZER_GRIDS)
        ),
    )
    parser.add_argument(
        "--streams",
        type=str,
        default=",".join(build_streams()),
        help="Comma-separated stream names.",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default=",".join(OPTIMIZER_GRIDS),
        help="Comma-separated optimizer names.",
    )
    args = parser.parse_args()
    normalizers = [n.strip() for n in args.normalizers.split(",") if n.strip()]
    stream_names = [s.strip() for s in args.streams.split(",") if s.strip()]
    optimizer_names = [o.strip() for o in args.optimizers.split(",") if o.strip()]
    main(
        output_dir=args.output_dir,
        seeds=args.seeds,
        burn_in=args.burn_in,
        measurement=args.measurement,
        normalizers=normalizers,
        stream_names=stream_names,
        optimizer_names=optimizer_names,
    )

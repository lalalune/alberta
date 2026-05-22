"""DoD-3: Pavlovian conditioning multi-seed suite.

Runs the full ACQUISITION -> EXTINCTION -> REACQUISITION sequence
through a Horde of multi-horizon prediction demons. Verifies the
classical animal-learning curves replicate with our framework.

Conditions
----------
- Horde with gamma in {0.0, 0.5, 0.9, 0.99}, lambda=0.0 (head-only).
- Linear Horde (`hidden_sizes=()`) -- the cleanest theoretical setting.
- 10 seeds.

Metrics
-------
For each demon, we measure mean prediction during the last 200 steps of
each phase. Acquisition: should be > 0. Extinction: should drop. Re-
acquisition: should recover.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    DemonType,
    GVFSpec,
    HordeLearner,
    blocking_scenario,
    create_horde_spec,
    reacquisition_scenario,
)

OUTPUT_DIR = Path("output/step3_dod3")
N_SEEDS = 10
GAMMAS = [0.0, 0.5, 0.9, 0.99]
N_ACQ = 2000
N_EXT = 2000
N_REACQ = 2000
N_BLOCK_PRETRAIN = 2000
N_BLOCK_COMPOUND = 2000


def collect_trajectory(stream: Any, key: Any, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    state = stream.init(key)
    obs_list, target_list = [], []
    for i in range(n_steps):
        ts, state = stream.step(state, jnp.array(i, dtype=jnp.int32))
        obs_list.append(np.asarray(ts.observation))
        target_list.append(float(jnp.squeeze(ts.target)))
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(target_list, dtype=np.float32),
    )


def mean_sem(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, sem


def paired_stats(a: list[float], b: list[float]) -> dict[str, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    mean = float(np.mean(diff))
    sem = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
    wins = int(np.sum(diff > 0.0))
    return {
        "mean_diff": mean,
        "sem_diff": sem,
        "wins": float(wins),
        "n": float(len(diff)),
    }


def train_horde_predictions(
    obs: np.ndarray, targets: np.ndarray, seed: int
) -> tuple[np.ndarray, HordeLearner, Any]:
    """Train the canonical linear Horde and return pre-update predictions."""
    feature_dim = obs.shape[1]
    demons = [
        GVFSpec(  # type: ignore[call-arg]
            name=f"d_g{g}",
            demon_type=DemonType.PREDICTION,
            gamma=g,
            lamda=0.0,
            cumulant_index=0,
        )
        for g in GAMMAS
    ]
    spec = create_horde_spec(demons)
    horde = HordeLearner(
        horde_spec=spec,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
    )
    state = horde.init(feature_dim, jr.key(seed + 100))
    cumulants = jnp.tile(jnp.asarray(targets)[:, None], (1, len(GAMMAS)))
    next_obs = jnp.concatenate([jnp.asarray(obs[1:]), jnp.asarray(obs[-1:])], axis=0)

    preds = np.zeros((len(obs), len(GAMMAS)), dtype=np.float32)
    for t in range(len(obs)):
        preds[t] = np.asarray(horde.predict(state, jnp.asarray(obs[t])))
        res = horde.update(
            state,
            jnp.asarray(obs[t]),
            cumulants[t],
            next_obs[t],
        )
        state = res.state
    return preds, horde, state


def run_reacquisition_seed(
    seed: int, n_acq: int, n_ext: int, n_reacq: int
) -> dict[str, float]:
    stream = reacquisition_scenario(
        n_acquisition=n_acq,
        n_extinction=n_ext,
        n_reacquisition=n_reacq,
        n_distractors=2,
        cs_us_delay=5,
        cs_duration=1,
        noise_std=0.02,
        distractor_prob=0.05,
    )

    obs, targets = collect_trajectory(stream, jr.key(seed), n_acq + n_ext + n_reacq)
    preds, _, _ = train_horde_predictions(obs, targets, seed)

    # Per-phase final-window means (last 200 steps per phase by default;
    # shorter for smoke runs).
    window = min(200, n_acq, n_ext, n_reacq)
    acq_end = n_acq
    ext_end = n_acq + n_ext
    reacq_end = ext_end + n_reacq

    out = {"seed": float(seed)}
    for k, g in enumerate(GAMMAS):
        acq_pred = float(np.mean(preds[acq_end - window:acq_end, k]))
        ext_pred = float(np.mean(preds[ext_end - window:ext_end, k]))
        reacq_pred = float(np.mean(preds[reacq_end - window:reacq_end, k]))
        out[f"acq_g{g}"] = acq_pred
        out[f"ext_g{g}"] = ext_pred
        out[f"reacq_g{g}"] = reacq_pred
    return out


def run_blocking_seed(seed: int, n_pretrain: int, n_compound: int) -> dict[str, float]:
    stream = blocking_scenario(
        n_pretrain=n_pretrain,
        n_compound=n_compound,
        n_distractors=2,
        cs_us_delay=5,
        cs_duration=1,
        noise_std=0.02,
        distractor_prob=0.05,
    )
    obs, targets = collect_trajectory(
        stream, jr.key(seed + 50_000), n_pretrain + n_compound
    )
    preds, horde, state = train_horde_predictions(obs, targets, seed + 50_000)

    window = 500
    compound = slice(n_pretrain, n_pretrain + n_compound)
    cs0_active = obs[compound, 0] > 0.5
    cs1_active = obs[compound, 1] > 0.5
    compound_onsets = np.where(cs0_active & cs1_active)[0] + n_pretrain
    tail = compound_onsets[compound_onsets >= (n_pretrain + n_compound - window)]
    if len(tail) == 0:
        tail = compound_onsets[-max(1, min(len(compound_onsets), 25)) :]

    out = {"seed": float(seed)}
    for k, g in enumerate(GAMMAS):
        compound_pred = float(np.mean(preds[tail, k]))
        cs0_obs = np.zeros(obs.shape[1], dtype=np.float32)
        cs1_obs = np.zeros(obs.shape[1], dtype=np.float32)
        cs0_obs[0] = 1.0
        cs1_obs[1] = 1.0
        cs0_pred = float(np.asarray(horde.predict(state, jnp.asarray(cs0_obs)))[k])
        cs1_pred = float(np.asarray(horde.predict(state, jnp.asarray(cs1_obs)))[k])
        out[f"block_cs0_g{g}"] = cs0_pred
        out[f"block_cs1_g{g}"] = cs1_pred
        out[f"block_compound_g{g}"] = compound_pred
        out[f"block_diff_g{g}"] = cs0_pred - cs1_pred
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--n-acq", type=int, default=N_ACQ)
    parser.add_argument("--n-ext", type=int, default=N_EXT)
    parser.add_argument("--n-reacq", type=int, default=N_REACQ)
    parser.add_argument("--n-block-pretrain", type=int, default=N_BLOCK_PRETRAIN)
    parser.add_argument("--n-block-compound", type=int, default=N_BLOCK_COMPOUND)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows: list[dict[str, float]] = []
    for seed in range(args.n_seeds):
        row = {
            **run_reacquisition_seed(seed, args.n_acq, args.n_ext, args.n_reacq),
            **run_blocking_seed(seed, args.n_block_pretrain, args.n_block_compound),
        }
        rows.append(row)
        # One-line summary per seed
        print(
            f"seed={seed:>2}  "
            + " ".join(
                f"g{g}: A={row[f'acq_g{g}']:.3f} E={row[f'ext_g{g}']:.3f} "
                f"R={row[f'reacq_g{g}']:.3f}"
                for g in GAMMAS
            )
        )

    csv_path = args.output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float]] = {}
    comparisons: dict[str, dict[str, float]] = {}
    for g in GAMMAS:
        a = [r[f"acq_g{g}"] for r in rows]
        e = [r[f"ext_g{g}"] for r in rows]
        rq = [r[f"reacq_g{g}"] for r in rows]
        b0 = [r[f"block_cs0_g{g}"] for r in rows]
        b1 = [r[f"block_cs1_g{g}"] for r in rows]
        a_mean, a_sem = mean_sem(a)
        e_mean, e_sem = mean_sem(e)
        rq_mean, rq_sem = mean_sem(rq)
        b0_mean, b0_sem = mean_sem(b0)
        b1_mean, b1_sem = mean_sem(b1)
        # Classical-conditioning curves: A > E (extinction reduces),
        # R > E (reacquisition recovers). Some γ's may be too short or
        # too long horizon to show much; this is descriptive.
        summary[f"gamma={g}"] = {
            "acq_mean": a_mean,
            "acq_sem": a_sem,
            "ext_mean": e_mean,
            "ext_sem": e_sem,
            "reacq_mean": rq_mean,
            "reacq_sem": rq_sem,
            "extinction_drop": a_mean - e_mean,
            "reacq_recovery": rq_mean - e_mean,
            "blocking_cs0_mean": b0_mean,
            "blocking_cs0_sem": b0_sem,
            "blocking_cs1_mean": b1_mean,
            "blocking_cs1_sem": b1_sem,
            "blocking_cs0_minus_cs1": b0_mean - b1_mean,
        }
        comparisons[f"gamma={g}_acq_minus_ext"] = paired_stats(a, e)
        comparisons[f"gamma={g}_reacq_minus_ext"] = paired_stats(rq, e)
        comparisons[f"gamma={g}_blocking_cs0_minus_cs1"] = paired_stats(b0, b1)

    t_total = time.perf_counter() - t0
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "comparisons": comparisons,
                "total_seconds": t_total,
                "n_seeds": args.n_seeds,
            },
            f,
            indent=2,
        )

    print("\n=== DoD-3 summary (per-phase mean prediction over last 200 steps) ===")
    print(
        f"{'gamma':>6}  {'A_mean':>7}  {'E_mean':>7}  {'R_mean':>7}  "
        f"{'A-E drop':>9}  {'R-E recov':>9}"
    )
    for g in GAMMAS:
        s = summary[f"gamma={g}"]
        print(
            f"{g:>6.2f}  {s['acq_mean']:>7.3f}  {s['ext_mean']:>7.3f}  "
            f"{s['reacq_mean']:>7.3f}  {s['extinction_drop']:>9.3f}  "
            f"{s['reacq_recovery']:>9.3f}"
        )
    print("\n=== DoD-3 blocking summary ===")
    print(f"{'gamma':>6}  {'CS0':>7}  {'CS1':>7}  {'CS0-CS1':>9}  {'wins':>7}")
    for g in GAMMAS:
        s = summary[f"gamma={g}"]
        c = comparisons[f"gamma={g}_blocking_cs0_minus_cs1"]
        print(
            f"{g:>6.2f}  {s['blocking_cs0_mean']:>7.3f}  "
            f"{s['blocking_cs1_mean']:>7.3f}  "
            f"{s['blocking_cs0_minus_cs1']:>9.3f}  "
            f"{int(c['wins']):>2}/{int(c['n']):<2}"
        )
    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Wrote {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

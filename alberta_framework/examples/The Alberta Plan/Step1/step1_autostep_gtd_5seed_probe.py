#!/usr/bin/env python3
"""5-seed Step 1 preliminary probe for Autostep-for-GTD(lambda).

Closes Alberta Plan footnote 11 by exposing Kearney et al. 2019's
Autostep-style normalized GTD(lambda) update as a public Step 1 baseline.
The supervised limit (gamma=0, lamda=0, rho=1) reduces to standard Autostep,
so this 5-seed x 5000-step probe simply confirms the optimizer behaves
correctly on the canonical AlbertaPlanStep1 stream alongside LMS and
Autostep, with results written to a JSON artifact for the docs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    LMS,
    Autostep,
    AutostepGTDLambda,
    EMANormalizer,
    LinearLearner,
)
from alberta_framework.streams.alberta_plan_step1 import AlbertaPlanStep1Stream

OUTPUT_DIR = (
    Path(__file__).resolve().parents[3] / "outputs" / "step1_canonical"
)
SEEDS = list(range(5))
BURN_IN = 1000
MEASUREMENT = 4000
TOTAL_STEPS = BURN_IN + MEASUREMENT


def build_stream() -> AlbertaPlanStep1Stream:
    return AlbertaPlanStep1Stream(
        feature_dim=20,
        num_relevant=5,
        drift_rate_w=0.001,
        drift_rate_b=0.001,
        noise_std=1.0,
    )


def per_seed_mse(learner: LinearLearner, stream: AlbertaPlanStep1Stream, key: Any) -> Any:
    learner_state = learner.init(stream.feature_dim)
    stream_state = stream.init(key)

    def step_fn(carry, idx):
        l_state, s_state = carry
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)
        return (result.state, new_s_state), result.metrics[0]

    (_, _), squared_errors = jax.lax.scan(
        step_fn, (learner_state, stream_state), jnp.arange(TOTAL_STEPS)
    )
    return jnp.mean(squared_errors[BURN_IN:])


def run_optimizer(name: str, optimizer: Any) -> dict[str, Any]:
    learner = LinearLearner(optimizer=optimizer, normalizer=EMANormalizer(decay=0.99))
    stream = build_stream()

    def per_seed(key):
        return per_seed_mse(learner, stream, key)

    keys = jnp.stack([jr.key(int(s)) for s in SEEDS])
    mses = jax.vmap(per_seed)(keys)
    mses.block_until_ready()
    arr = np.asarray(mses, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.inf)
    return {
        "optimizer": name,
        "per_seed": [float(v) for v in arr.tolist()],
        "mean_mse": float(np.mean(arr[np.isfinite(arr)])),
        "stderr": float(np.std(arr[np.isfinite(arr)], ddof=1) / np.sqrt(len(arr))),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    optimizers = {
        "LMS": LMS(step_size=0.01),
        "Autostep": Autostep(initial_step_size=0.02, meta_step_size=0.01, tau=10000.0),
        "AutostepGTDLambda": AutostepGTDLambda(
            initial_step_size=0.02, meta_step_size=0.01, tau=10000.0, trace_decay=0.0
        ),
    }
    results = {name: run_optimizer(name, opt) for name, opt in optimizers.items()}
    wall_clock_s = time.time() - t0

    payload = {
        "config": {
            "stream": "AlbertaPlanStep1",
            "feature_dim": 20,
            "num_relevant": 5,
            "drift_rate_w": 0.001,
            "drift_rate_b": 0.001,
            "noise_std": 1.0,
            "normalizer": "EMA(decay=0.99)",
            "seeds": SEEDS,
            "burn_in_steps": BURN_IN,
            "measurement_steps": MEASUREMENT,
        },
        "wall_clock_s": wall_clock_s,
        "results": results,
        "notes": (
            "Preliminary 5-seed probe. The Step 1 supervised limit of "
            "Autostep-for-GTD(lambda) (Kearney et al. 2019, gamma=lambda=0, "
            "rho=1) reduces algebraically to Autostep, so the two columns are "
            "expected to coincide within numerical noise. Use the full 30-seed "
            "step1_full_baselines.py sweep for paper-grade evidence."
        ),
    }

    out_json = OUTPUT_DIR / "autostep_gtd_5seed_results.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")

    summary_lines: list[str] = []
    summary_lines.append("# Autostep-for-GTD(lambda) — 5-seed Preliminary Probe\n")
    summary_lines.append(
        "Preliminary 5-seed evidence that the new ``AutostepGTDLambda`` "
        "optimizer (Kearney et al. 2019, supervised limit) operates correctly "
        "on the canonical Alberta Plan Step 1 stream. The supervised limit "
        "with ``gamma=lambda=0`` reduces algebraically to Autostep, so the two "
        "columns should match within numerical noise.\n"
    )
    summary_lines.append("## Configuration\n")
    summary_lines.append(f"- Seeds: {SEEDS}")
    summary_lines.append(f"- Burn-in steps: {BURN_IN:,}")
    summary_lines.append(f"- Measurement steps: {MEASUREMENT:,}")
    summary_lines.append("- Stream: AlbertaPlanStep1Stream(20, num_relevant=5)")
    summary_lines.append("- Normalizer: EMA(decay=0.99)\n")
    summary_lines.append("## Results (mean MSE on tail, mean ± stderr over 5 seeds)\n")
    summary_lines.append("| Optimizer | mean MSE | stderr |")
    summary_lines.append("|---|---|---|")
    for name, stats in results.items():
        summary_lines.append(
            f"| {name} | {stats['mean_mse']:.4f} | {stats['stderr']:.4f} |"
        )
    summary_lines.append("")
    summary_lines.append(f"Wall-clock: {wall_clock_s:.2f} s\n")
    summary_lines.append(
        "**Status:** 5-seed preliminary evidence. For paper-grade headline "
        "claims, run the 30-seed, joint normalizer/optimizer-grid sweep in "
        "``step1_full_baselines.py``."
    )
    out_md = OUTPUT_DIR / "autostep_gtd_5seed_SUMMARY.md"
    out_md.write_text("\n".join(summary_lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

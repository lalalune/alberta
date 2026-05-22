"""DoD-7: Feature finding under TD targets multi-seed sweep.

Tests whether ``CumulantDiscovery`` (Phase F) can keep candidate
cumulants with persistent transition signal under the GVF convention
``c_{t+1}``.

Setup
-----
- 8-dim temporally correlated observation stream.
- The first two channels carry slow latent dynamics with larger variance;
  the remaining channels are weaker, faster nuisance signals.
- 16 cumulant candidates, each a random unit projection.
- Surprise EMA (decay=0.99); replacement_rate=0.01; maturity=200.
- 5000 steps per seed; 10 seeds.
- Conditions:
  * discovery_on  (replacement enabled)
  * discovery_off (no replacement -- baseline of random projections)
- Metric: median final utility across surviving candidates.
  Higher utility means the surviving set captures harder-to-predict
  cumulants. We also track how often the SAME projection appears in
  both conditions (it shouldn't, by design of replacement).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import CumulantDiscovery

OUTPUT_DIR = Path("output/step3_dod7")
N_SEEDS = 10
N_STEPS = 5000
RAW_DIM = 8
N_CANDIDATES = 16


def run(seed: int, enabled: bool) -> dict[str, float]:
    d = CumulantDiscovery(
        raw_dim=RAW_DIM,
        n_candidates=N_CANDIDATES,
        decay_rate=0.99,
        replacement_rate=0.01,
        maturity_threshold=200,
        predictor_step_size=0.05,
        gamma=0.0,
        enabled=enabled,
    )
    state = d.init(jr.key(seed))

    rng = np.random.default_rng(seed)
    n_replacements = 0
    initial_proj = np.asarray(state.projections).copy()
    obs_np = rng.normal(size=RAW_DIM).astype(np.float32)
    ar = np.array([0.97, 0.94, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15], dtype=np.float32)
    innovation_scale = np.array(
        [0.45, 0.35, 0.08, 0.08, 0.06, 0.06, 0.04, 0.04], dtype=np.float32
    )

    for _ in range(N_STEPS):
        innovation = rng.normal(size=RAW_DIM).astype(np.float32) * innovation_scale
        next_obs_np = ar * obs_np + innovation
        obs = jnp.asarray(obs_np)
        next_obs = jnp.asarray(next_obs_np)
        prev_proj = np.asarray(state.projections).copy()
        state = d.step(state, obs, next_obs)
        new_state = d.maybe_replace(state)
        if not np.allclose(np.asarray(new_state.projections), prev_proj):
            n_replacements += 1
        state = new_state
        obs_np = next_obs_np

    final_proj = np.asarray(state.projections)
    final_util = np.asarray(state.utility)

    # Fraction of candidates that have been replaced relative to start
    n_changed = int(
        np.sum(
            ~np.all(np.isclose(final_proj, initial_proj, atol=1e-6), axis=1)
        )
    )

    return {
        "seed": seed,
        "enabled": enabled,
        "median_utility": float(np.median(final_util)),
        "min_utility": float(np.min(final_util)),
        "max_utility": float(np.max(final_util)),
        "n_replacements_observed": float(n_replacements),
        "n_changed_projections": float(n_changed),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows: list[dict[str, float]] = []
    for enabled in [False, True]:
        for seed in range(N_SEEDS):
            row = run(seed, enabled)
            rows.append(row)
            print(
                f"enabled={str(enabled):>5} seed={seed:>2}  "
                f"med_util={row['median_utility']:.4f}  "
                f"replacements={int(row['n_replacements_observed'])}  "
                f"changed={int(row['n_changed_projections'])}"
            )

    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float]] = {}
    for enabled in [False, True]:
        sub = [r for r in rows if r["enabled"] == enabled]
        utils_med = [r["median_utility"] for r in sub]
        replacements = [r["n_replacements_observed"] for r in sub]
        summary[f"enabled={enabled}"] = {
            "median_utility_mean": float(np.mean(utils_med)),
            "median_utility_std": float(np.std(utils_med)),
            "n_replacements_mean": float(np.mean(replacements)),
            "n_changed_mean": float(np.mean([r["n_changed_projections"] for r in sub])),
        }

    t_total = time.perf_counter() - t0
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "total_seconds": t_total}, f, indent=2)

    print("\n=== DoD-7 summary ===")
    for k, s in summary.items():
        print(
            f"  {k:>14}  med_util={s['median_utility_mean']:.4f}±{s['median_utility_std']:.4f}  "
            f"replacements={s['n_replacements_mean']:.1f}  "
            f"changed={s['n_changed_mean']:.1f}/{N_CANDIDATES}"
        )
    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Wrote {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

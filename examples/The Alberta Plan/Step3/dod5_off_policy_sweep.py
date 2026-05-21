"""DoD-5: Off-policy convergence multi-seed sweep.

Tests `OffPolicyTDLinearLearner` on a small bandit-with-state where the
target policy differs from the behavior policy. Measures convergence to
the true target-policy V across multiple seeds.

Setup
-----
- 4-state random walk with terminal rewards 0 (left) / 1 (right).
- Behavior policy: uniform random.
- Target policy: always go right => true V = 1 for every state.
- Off-policy IS ratios: rho = 1 / 0.5 = 2 if action=right, 0 if action=left.

Conditions:
- naive IS (clip = 1000)
- Retrace clip=1
- Retrace clip=2
- Per-decision IS (clip = inf)

Output: ``output/step3_dod5/results.csv`` with per-seed final RMSE.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from alberta_framework import OffPolicyTDLinearLearner

OUTPUT_DIR = Path("output/step3_dod5")
N_SEEDS = 12
N_STATES = 4
N_EPISODES = 2000
ALPHA = 0.05


def generate_episode(rng: np.random.Generator) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    eye = np.eye(N_STATES, dtype=np.float32)
    state = N_STATES // 2
    obs_l, nxt_l, rew_l, gam_l, act_l = [], [], [], [], []
    while True:
        action = int(rng.integers(0, 2))
        new_state = state - 1 if action == 0 else state + 1
        obs_l.append(eye[state])
        act_l.append(action)
        if new_state < 0:
            rew_l.append(0.0)
            nxt_l.append(np.zeros(N_STATES, dtype=np.float32))
            gam_l.append(0.0)
            break
        if new_state >= N_STATES:
            rew_l.append(1.0)
            nxt_l.append(np.zeros(N_STATES, dtype=np.float32))
            gam_l.append(0.0)
            break
        rew_l.append(0.0)
        nxt_l.append(eye[new_state])
        gam_l.append(1.0)
        state = new_state
    return (
        np.asarray(obs_l), np.asarray(rew_l, dtype=np.float32),
        np.asarray(nxt_l), np.asarray(gam_l, dtype=np.float32),
        np.asarray(act_l),
    )


def run_one_seed(seed: int, retrace_clip: float) -> dict[str, float]:
    learner = OffPolicyTDLinearLearner(
        step_size=ALPHA, trace_decay=0.0, retrace_clip=retrace_clip
    )
    state = learner.init(N_STATES)
    rng = np.random.default_rng(seed)
    for _ in range(N_EPISODES):
        obs, rew, nxt, gam, actions = generate_episode(rng)
        for t in range(len(rew)):
            rho = 2.0 if actions[t] == 1 else 0.0
            res = learner.update(
                state,
                jnp.asarray(obs[t]),
                jnp.asarray(rew[t]),
                jnp.asarray(nxt[t]),
                jnp.asarray(gam[t]),
                jnp.float32(rho),
            )
            state = res.state

    eye = np.eye(N_STATES, dtype=np.float32)
    v_est = np.array(
        [
            float(jnp.dot(state.weights, jnp.asarray(eye[s])) + state.bias)
            for s in range(N_STATES)
        ]
    )
    v_true = np.ones(N_STATES, dtype=np.float32)
    rmse = float(np.sqrt(np.mean((v_est - v_true) ** 2)))
    return {
        "seed": seed,
        "retrace_clip": retrace_clip,
        "rmse": rmse,
        "v_estimated_max_abs": float(np.max(np.abs(v_est))),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows: list[dict[str, float]] = []
    for clip in [1.0, 2.0, 1000.0, float("inf")]:
        for seed in range(N_SEEDS):
            row = run_one_seed(seed, clip)
            rows.append(row)
            print(
                f"clip={clip:>5} seed={seed:>2} rmse={row['rmse']:.4f} "
                f"|V|max={row['v_estimated_max_abs']:.3f}"
            )

    t_total = time.perf_counter() - t0

    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate per clip
    summary: dict[str, dict[str, float]] = {}
    for clip in [1.0, 2.0, 1000.0, float("inf")]:
        clip_rows = [r for r in rows if r["retrace_clip"] == clip]
        rmses = [r["rmse"] for r in clip_rows]
        max_abs = [r["v_estimated_max_abs"] for r in clip_rows]
        diverged = sum(1 for v in max_abs if v > 100.0)
        summary[str(clip)] = {
            "n_seeds": len(clip_rows),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "rmse_median": float(np.median(rmses)),
            "n_diverged": diverged,
            "max_abs_v_max": float(np.max(max_abs)),
        }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"summary": summary, "total_seconds": t_total, "n_seeds": N_SEEDS},
            f,
            indent=2,
        )

    print("\n=== DoD-5 summary ===")
    for clip, s in summary.items():
        print(
            f"clip={clip:>10}  RMSE mean={s['rmse_mean']:.4f}  "
            f"std={s['rmse_std']:.4f}  diverged={s['n_diverged']}/{s['n_seeds']}"
        )
    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Wrote {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

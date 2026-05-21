"""DoD-9: Critic-for-control capstone sweep.

Tests whether auxiliary GVF prediction demons (the Step 3 critic) help
SARSA on a continuing control problem. Three conditions:

  1. SARSA-only baseline (no auxiliary tasks).
  2. SARSA + prediction-Horde (auxiliary GVF demons sharing the same
     trunk; predict reward at multiple horizons).
  3. SARSA + prediction-Horde + CBP + history features (Phases D/G).

Setup
-----
- Continuing GridWorld-like task implemented inline:
  * 8-dim observation: one-hot state in a 4x4 toroidal grid.
  * 4 actions (up/down/left/right) move on the torus.
  * Reward: +1 in one fixed cell, 0 elsewhere; agent teleports back
    to a random cell after collecting reward.
  * Continuing: no terminal state.
- Defaults: 8 seeds each, 30000 steps per seed.

Metric: total reward in the last 5000 steps. Higher is better.
"""

from __future__ import annotations

import csv
import json
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    CBPMultiHeadMLPLearner,
    ContinualBackpropConfig,
    DemonType,
    GVFSpec,
    HistoryFeatureExtractor,
    SARSAAgent,
    SARSAConfig,
)

OUTPUT_DIR = Path("output/step3_dod9")
N_SEEDS = 8
N_STEPS = 30_000
LAST_WINDOW = 5_000
GRID = 4  # 4x4 grid -> 16 states
N_ACTIONS = 4
GAMMA_RL = 0.95
ALPHA = 0.1
ACTION_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
GOAL_CELL = (3, 3)


CONDITIONS = [
    ("baseline_sarsa", False, False, False),
    ("sarsa_prediction_horde", True, False, False),
    ("sarsa_horde_cbp_history", True, True, True),
]


def state_to_obs(r: int, c: int) -> np.ndarray:
    """Concatenated one-hot row and column: dim = 2 * GRID = 8."""
    obs = np.zeros(2 * GRID, dtype=np.float32)
    obs[r] = 1.0
    obs[GRID + c] = 1.0
    return obs


def step_env(rng: np.random.Generator, r: int, c: int, action: int) -> tuple[int, int, float]:
    dr, dc = ACTION_DELTAS[action]
    nr = (r + dr) % GRID
    nc = (c + dc) % GRID
    reward = 0.0
    if (nr, nc) == GOAL_CELL:
        reward = 1.0
        # Teleport on collection
        nr = int(rng.integers(0, GRID))
        nc = int(rng.integers(0, GRID))
    return nr, nc, reward


def run_one(
    seed: int,
    use_aux_horde: bool,
    use_history: bool,
    use_cbp: bool,
    n_steps: int,
    last_window_size: int,
    hidden_sizes: tuple[int, ...],
    agent: SARSAAgent | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    # Build optional history feature extractor
    if use_history:
        extractor = HistoryFeatureExtractor(
            raw_dim=2 * GRID, decay_rates=(0.5, 0.9), include_raw=True
        )
        h_state = extractor.init()
        feat_dim = extractor.feature_dim()
    else:
        extractor = None
        h_state = None
        feat_dim = 2 * GRID

    # Build SARSA agent (with optional auxiliary prediction demons). The sweep
    # passes a prebuilt agent so JIT compilation is reused across seeds.
    # Keeping construction here as a fallback preserves the single-run API.
    aux_demons: list[GVFSpec] = []
    if use_aux_horde:
        # Predict reward at gamma in {0, 0.5, 0.9}
        for g in [0.0, 0.5, 0.9]:
            aux_demons.append(
                GVFSpec(  # type: ignore[call-arg]
                    name=f"aux_g{g}",
                    demon_type=DemonType.PREDICTION,
                    gamma=g,
                    lamda=0.0,
                    cumulant_index=N_ACTIONS,  # auxiliary cumulant (reward)
                )
            )

    if agent is None:
        agent = build_agent(
            use_aux_horde=use_aux_horde,
            use_cbp=use_cbp,
            n_steps=n_steps,
            hidden_sizes=hidden_sizes,
            aux_demons=aux_demons,
        )
    state = agent.init(feat_dim, jr.key(seed + 7000))

    r, c = int(rng.integers(0, GRID)), int(rng.integers(0, GRID))
    if extractor is not None:
        obs_aug, h_state = extractor.step(h_state, jnp.asarray(state_to_obs(r, c)))
    else:
        obs_aug = jnp.asarray(state_to_obs(r, c))

    rewards_per_step = np.zeros(n_steps, dtype=np.float32)

    # Bootstrap: pick first action and seed last_action / last_observation
    action, new_key = agent.select_action(state, obs_aug)
    state = state.replace(  # type: ignore[attr-defined]
        last_action=action,
        last_observation=obs_aug,
        rng_key=new_key,
    )

    for t in range(n_steps):
        nr, nc, reward = step_env(rng, r, c, int(action))
        rewards_per_step[t] = reward
        next_obs_raw = state_to_obs(nr, nc)
        if extractor is not None:
            next_obs_aug, h_state_new = extractor.step(
                h_state, jnp.asarray(next_obs_raw)
            )
        else:
            next_obs_aug = jnp.asarray(next_obs_raw)
            h_state_new = None

        # Pick next action a' (on-policy)
        next_action, new_key = agent.select_action(state, next_obs_aug)
        state = state.replace(rng_key=new_key)

        # Auxiliary cumulants: aux demons predict reward
        if aux_demons:
            aux_cumulants = jnp.full(len(aux_demons), reward, dtype=jnp.float32)
        else:
            aux_cumulants = None

        result = agent.update(
            state,
            jnp.float32(reward),
            next_obs_aug,
            jnp.float32(0.0),  # never terminated in this continuing task
            next_action,
            prediction_cumulants=aux_cumulants,
        )
        state = result.state

        # Advance
        r, c = nr, nc
        obs_aug = next_obs_aug
        if h_state_new is not None:
            h_state = h_state_new
        action = next_action

    last_window = int(np.sum(rewards_per_step[-last_window_size:]))
    total = int(np.sum(rewards_per_step))
    return {
        "seed": seed,
        "use_aux_horde": int(use_aux_horde),
        "use_history": int(use_history),
        "use_cbp": int(use_cbp),
        "total_reward": total,
        "last_window_reward": last_window,
        "first_window_reward": int(np.sum(rewards_per_step[:last_window_size])),
    }


def build_agent(
    use_aux_horde: bool,
    use_cbp: bool,
    n_steps: int,
    hidden_sizes: tuple[int, ...],
    aux_demons: list[GVFSpec] | None = None,
) -> SARSAAgent:
    """Build one reusable agent instance for a sweep condition."""
    if aux_demons is None:
        aux_demons = []
        if use_aux_horde:
            for g in [0.0, 0.5, 0.9]:
                aux_demons.append(
                    GVFSpec(  # type: ignore[call-arg]
                        name=f"aux_g{g}",
                        demon_type=DemonType.PREDICTION,
                        gamma=g,
                        lamda=0.0,
                        cumulant_index=N_ACTIONS,
                    )
                )

    config = SARSAConfig(  # type: ignore[call-arg]
        n_actions=N_ACTIONS,
        gamma=GAMMA_RL,
        epsilon_start=0.30,
        epsilon_end=0.05,
        epsilon_decay_steps=n_steps // 2,
    )
    agent = SARSAAgent(
        sarsa_config=config,
        prediction_demons=aux_demons if aux_demons else None,
        hidden_sizes=hidden_sizes,
        step_size=ALPHA,
        sparsity=0.0,
        use_layer_norm=bool(hidden_sizes),
    )
    if use_cbp:
        # SARSAAgent intentionally exposes Horde as the control substrate.
        # For this capstone condition, reuse the same Horde spec and swap in
        # the existing CBP multi-head learner so control and prediction demons
        # still share one trunk while CBP manages hidden-unit replacement.
        per_head_gl = tuple(float(d.gamma * d.lamda) for d in agent.horde.horde_spec.demons)
        cast(Any, agent.horde)._learner = CBPMultiHeadMLPLearner(  # noqa: SLF001
            n_heads=agent.horde.n_demons,
            hidden_sizes=hidden_sizes,
            cbp_config=ContinualBackpropConfig(  # type: ignore[call-arg]
                enabled=bool(hidden_sizes),
                decay_rate=0.99,
                replacement_rate=1e-4,
                maturity_threshold=100,
            ),
            step_size=ALPHA,
            sparsity=0.0,
            use_layer_norm=bool(hidden_sizes),
            per_head_gamma_lamda=per_head_gl,
        )
    return agent


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--last-window", type=int, default=LAST_WINDOW)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Single shared-trunk hidden layer width. Use 0 for linear heads.",
    )
    return parser


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    """Return the condition/seed key for a result row."""
    return str(row["condition"]), int(row["seed"])


def _load_existing_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Load existing result rows so interrupted runs can resume."""
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "condition": row["condition"],
                    "use_aux_horde": int(row["use_aux_horde"]),
                    "use_history": int(row["use_history"]),
                    "use_cbp": int(row["use_cbp"]),
                    "total_reward": int(row["total_reward"]),
                    "last_window_reward": int(row["last_window_reward"]),
                    "first_window_reward": int(row["first_window_reward"]),
                }
            )
    return rows


def _write_results(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write result rows in stable condition/seed order."""
    ordered = sorted(rows, key=_row_key)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "condition", "use_aux_horde", "use_history", "use_cbp",
                "total_reward", "last_window_reward", "first_window_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(ordered)


def _write_summary(
    summary_path: Path,
    rows: list[dict[str, Any]],
    *,
    elapsed_s: float,
    n_seeds: int,
    steps: int,
    last_window_size: int,
    hidden_sizes: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    """Write the aggregate JSON summary and return the summary mapping."""
    summary: dict[str, dict[str, float]] = {}
    for label, _, _, _ in CONDITIONS:
        sub = [r for r in rows if r["condition"] == label and int(r["seed"]) < n_seeds]
        last_window = [r["last_window_reward"] for r in sub]
        total = [r["total_reward"] for r in sub]
        summary[label] = {
            "last_window_reward_mean": float(np.mean(last_window)) if last_window else float("nan"),
            "last_window_reward_std": float(np.std(last_window)) if last_window else float("nan"),
            "total_reward_mean": float(np.mean(total)) if total else float("nan"),
            "total_reward_std": float(np.std(total)) if total else float("nan"),
            "n_seeds": len(sub),
        }

    with open(summary_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "total_seconds": elapsed_s,
                "config": {
                    "n_seeds": n_seeds,
                    "steps": steps,
                    "last_window": last_window_size,
                    "hidden_sizes": list(hidden_sizes),
                    "conditions": [c[0] for c in CONDITIONS],
                    "resumable": True,
                },
            },
            f,
            indent=2,
        )
    return summary


def main() -> int:
    args = parse_args().parse_args()
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.last_window <= 0:
        raise ValueError("--last-window must be positive")
    if args.hidden_size < 0:
        raise ValueError("--hidden-size must be non-negative")

    last_window_size = min(args.last_window, args.steps)
    hidden_sizes = () if args.hidden_size == 0 else (args.hidden_size,)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    summary_path = args.output_dir / "summary.json"
    t0 = time.perf_counter()
    rows = _load_existing_rows(csv_path)
    completed = {_row_key(row) for row in rows}
    for label, aux, hist, cbp in CONDITIONS:
        agent = build_agent(
            use_aux_horde=aux,
            use_cbp=cbp,
            n_steps=args.steps,
            hidden_sizes=hidden_sizes,
        )
        for seed in range(args.n_seeds):
            if (label, seed) in completed:
                print(f"[{label:>22}] seed={seed:>2}  already complete; skipping")
                continue
            t_run = time.perf_counter()
            row = run_one(
                seed=seed,
                use_aux_horde=aux,
                use_history=hist,
                use_cbp=cbp,
                n_steps=args.steps,
                last_window_size=last_window_size,
                hidden_sizes=hidden_sizes,
                agent=agent,
            )
            row["condition"] = label
            rt = time.perf_counter() - t_run
            rows.append(row)
            completed.add((label, seed))
            _write_results(csv_path, rows)
            _write_summary(
                summary_path,
                rows,
                elapsed_s=time.perf_counter() - t0,
                n_seeds=args.n_seeds,
                steps=args.steps,
                last_window_size=last_window_size,
                hidden_sizes=hidden_sizes,
            )
            print(
                f"[{label:>22}] seed={seed:>2}  "
                f"first_window={row['first_window_reward']:>4}  "
                f"last_window={row['last_window_reward']:>4}  "
                f"total={row['total_reward']:>5}  ({rt:.1f}s)"
            )

    t_total = time.perf_counter() - t0
    _write_results(csv_path, rows)
    summary = _write_summary(
        summary_path,
        rows,
        elapsed_s=t_total,
        n_seeds=args.n_seeds,
        steps=args.steps,
        last_window_size=last_window_size,
        hidden_sizes=hidden_sizes,
    )

    print("\n=== DoD-9 summary (continuing GridWorld torus) ===")
    print(
        f"  {'condition':<25} {'last_mean':>12} {'last_std':>12} "
        f"{'total_mean':>10}"
    )
    for label, s in summary.items():
        print(
            f"  {label:<25} {s['last_window_reward_mean']:>12.1f} "
            f"{s['last_window_reward_std']:>12.1f} {s['total_reward_mean']:>10.1f}"
        )
    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Wrote {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

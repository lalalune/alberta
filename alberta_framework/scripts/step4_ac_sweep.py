# ruff: noqa: E402
"""Step 4 actor-critic hyperparameter sweep on cartpole/0.

Runs a sparse Latin-hypercube subset over (temperature, actor_lamda,
critic_step_size, actor_step_size, kappa) and reports the mean episode return
over the last K episodes per seed. Designed to fit in ~10-15 minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from benchmarks.bsuite._bsuite_path import add_bsuite_to_path  # noqa: E402

add_bsuite_to_path()
import bsuite  # noqa: E402

from benchmarks.bsuite.agents import actor_critic  # noqa: E402
from benchmarks.bsuite.wrappers import ContinuingWrapper  # noqa: E402


def run_one(
    seed: int,
    num_steps: int,
    config: dict[str, Any],
) -> tuple[float, list[float]]:
    """Run one (config, seed) trial. Returns (mean_recent_episode_return, all_episode_returns)."""
    raw_env = bsuite.load_from_id("cartpole/0")
    env = ContinuingWrapper(raw_env, mode="continuing")
    agent = actor_critic.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        seed=seed,
        **config,
    )
    timestep = env.reset()
    episode_returns: list[float] = []
    current_return = 0.0
    for _ in range(num_steps):
        action = agent.select_action(timestep)
        new_timestep = env.step(action)
        agent.update(timestep, action, new_timestep)
        reward = float(new_timestep.reward) if new_timestep.reward is not None else 0.0
        current_return += reward
        # Episode boundary detected via discount=0
        if float(new_timestep.discount) == 0.0:
            episode_returns.append(current_return)
            current_return = 0.0
        timestep = new_timestep
    if not episode_returns:
        return current_return, [current_return]
    # Recent half average
    half = max(1, len(episode_returns) // 2)
    return float(np.mean(episode_returns[-half:])), episode_returns


def lhs_configs() -> list[dict[str, Any]]:
    """Build a sparse Latin-hypercube sample of configs."""
    base = {
        "implementation": "horde_core",
        "optimizer_name": "autostep",
        "initial_step_size": 0.01,
        "meta_step_size": 0.01,
        "tau": 10000.0,
        "normalizer_decay": 0.99,
        "discount": 0.99,
        "critic_lamda": 0.0,
        "n_auxiliary_demons": 0,
        "sparsity": 0.0,
        "hidden_sizes": (64, 64),
    }
    # 21 configs total — current default + sparse LHS over 5 dimensions
    grid: list[dict[str, Any]] = [
        # CURRENT DEFAULT
        {"temperature": 1.0, "actor_lamda": 0.9, "actor_step_size": 0.03, "kappa": 2.0,
         "_initial_step_size": 0.01, "_label": "default"},
        # Lower temperature variants (sharper policy)
        {"temperature": 0.5, "actor_lamda": 0.9, "actor_step_size": 0.03, "kappa": 2.0,
         "_initial_step_size": 0.01, "_label": "temp0.5"},
        {"temperature": 0.5, "actor_lamda": 0.5, "actor_step_size": 0.1, "kappa": 2.0,
         "_initial_step_size": 0.03, "_label": "temp0.5_alpha0.1"},
        {"temperature": 0.5, "actor_lamda": 0.0, "actor_step_size": 0.1, "kappa": None,
         "_initial_step_size": 0.03, "_label": "temp0.5_nokappa"},
        # Higher temperature variants
        {"temperature": 2.0, "actor_lamda": 0.9, "actor_step_size": 0.03, "kappa": 2.0,
         "_initial_step_size": 0.01, "_label": "temp2.0"},
        {"temperature": 2.0, "actor_lamda": 0.0, "actor_step_size": 0.1, "kappa": 2.0,
         "_initial_step_size": 0.03, "_label": "temp2.0_lam0_alpha0.1"},
        # Stronger actor learning
        {"temperature": 1.0, "actor_lamda": 0.0, "actor_step_size": 0.1, "kappa": 2.0,
         "_initial_step_size": 0.03, "_label": "no_lam_strong_actor"},
        {"temperature": 1.0, "actor_lamda": 0.5, "actor_step_size": 0.1, "kappa": 2.0,
         "_initial_step_size": 0.03, "_label": "midlam_strong_actor"},
        {"temperature": 1.0, "actor_lamda": 0.9, "actor_step_size": 0.1, "kappa": 2.0,
         "_initial_step_size": 0.03, "_label": "lam0.9_strong_actor"},
        {"temperature": 1.0, "actor_lamda": 0.9, "actor_step_size": 0.1, "kappa": 0.5,
         "_initial_step_size": 0.03, "_label": "lam0.9_strong_actor_lowkappa"},
        # No bounder
        {"temperature": 1.0, "actor_lamda": 0.9, "actor_step_size": 0.1, "kappa": None,
         "_initial_step_size": 0.03, "_label": "lam0.9_strong_actor_nokappa"},
        # Strong critic
        {"temperature": 1.0, "actor_lamda": 0.9, "actor_step_size": 0.03, "kappa": 2.0,
         "_initial_step_size": 0.1, "_label": "strong_critic"},
        {"temperature": 1.0, "actor_lamda": 0.5, "actor_step_size": 0.03, "kappa": 2.0,
         "_initial_step_size": 0.1, "_label": "strong_critic_midlam"},
        # Combinations
        {"temperature": 0.5, "actor_lamda": 0.5, "actor_step_size": 0.03, "kappa": 0.5,
         "_initial_step_size": 0.03, "_label": "temp0.5_lowkappa"},
        {"temperature": 1.0, "actor_lamda": 0.0, "actor_step_size": 0.03, "kappa": None,
         "_initial_step_size": 0.03, "_label": "no_trace_no_kappa"},
        {"temperature": 0.7, "actor_lamda": 0.7, "actor_step_size": 0.05, "kappa": 1.0,
         "_initial_step_size": 0.05, "_label": "midmix"},
        {"temperature": 1.5, "actor_lamda": 0.9, "actor_step_size": 0.1, "kappa": 0.5,
         "_initial_step_size": 0.03, "_label": "temp1.5_lam0.9"},
        {"temperature": 0.5, "actor_lamda": 0.9, "actor_step_size": 0.03, "kappa": None,
         "_initial_step_size": 0.01, "_label": "temp0.5_nokappa_default"},
        {"temperature": 1.0, "actor_lamda": 0.9, "actor_step_size": 0.05, "kappa": 1.0,
         "_initial_step_size": 0.03, "_label": "midkappa_midactor"},
        {"temperature": 0.7, "actor_lamda": 0.5, "actor_step_size": 0.1, "kappa": 2.0,
         "_initial_step_size": 0.1, "_label": "strong_both"},
        {"temperature": 1.0, "actor_lamda": 0.5, "actor_step_size": 0.05, "kappa": 1.0,
         "_initial_step_size": 0.05, "_label": "balanced"},
    ]
    configs = []
    for entry in grid:
        cfg = dict(base)
        cfg.update(
            {
                "temperature": entry["temperature"],
                "actor_lamda": entry["actor_lamda"],
                "actor_step_size": entry["actor_step_size"],
                "initial_step_size": entry["_initial_step_size"],
            }
        )
        if entry["kappa"] is None:
            # Disable bounder by setting very large kappa
            cfg["kappa"] = 1e9
        else:
            cfg["kappa"] = entry["kappa"]
        cfg["_label"] = entry["_label"]
        configs.append(cfg)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--output", type=str, default="outputs/step4_ac_sweep/sweep_results.json")
    args = parser.parse_args()

    configs = lhs_configs()
    results = []
    t_start = time.time()
    for ci, cfg in enumerate(configs):
        label = cfg.pop("_label")
        seed_returns: list[float] = []
        seed_curves: list[list[float]] = []
        cfg_t = time.time()
        for seed in range(args.num_seeds):
            mean_recent, returns = run_one(seed, args.num_steps, cfg)
            seed_returns.append(mean_recent)
            seed_curves.append(returns)
        elapsed = time.time() - cfg_t
        mean = float(np.mean(seed_returns))
        std = float(np.std(seed_returns))
        print(
            f"[{ci+1}/{len(configs)}] {label:30s} mean_recent={mean:.2f} ± {std:.2f}  "
            f"({elapsed:.1f}s, total {time.time()-t_start:.0f}s)"
        )
        results.append(
            {
                "label": label,
                "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                "seed_recent_returns": seed_returns,
                "mean_recent": mean,
                "std_recent": std,
                "all_returns": seed_curves,
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    # Summary table sorted by mean_recent
    print("\n\n=== Sorted summary (by mean_recent return) ===")
    for r in sorted(results, key=lambda x: -x["mean_recent"]):
        print(f"  {r['label']:30s} {r['mean_recent']:6.2f} ± {r['std_recent']:5.2f}")


if __name__ == "__main__":
    main()

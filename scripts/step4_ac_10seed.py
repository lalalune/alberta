# ruff: noqa: E402
"""Step 4 actor-critic 10-seed head-to-head: tuned vs default vs SARSA vs Q.

Runs cartpole/0 (episode_return) and catch/0 (total_regret) for the four
configurations the diagnosis report references.
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

from benchmarks.bsuite.agents import (  # noqa: E402
    actor_critic,
    autostep_dqn,
)
from benchmarks.bsuite.agents import (
    sarsa as sarsa_agent,
)
from benchmarks.bsuite.wrappers import ContinuingWrapper  # noqa: E402

# Default actor-critic config
DEFAULT_AC = dict(
    implementation="horde_core",
    optimizer_name="autostep",
    initial_step_size=0.01,
    meta_step_size=0.01,
    tau=10000.0,
    kappa=2.0,
    normalizer_decay=0.99,
    discount=0.99,
    temperature=1.0,
    actor_step_size=0.03,
    actor_lamda=0.9,
    critic_lamda=0.0,
    hidden_sizes=(64, 64),
)

# Tuned actor-critic config (winner of cartpole sweep, low variance)
TUNED_AC = dict(DEFAULT_AC)
TUNED_AC.update(
    dict(
        temperature=0.5,
        actor_lamda=0.9,
        actor_step_size=0.03,
        kappa=1e9,  # disable bounder
    )
)

# Q-learning (autostep_dqn) defaults — match existing config
DEFAULT_QL = dict(
    initial_step_size=0.01,
    meta_step_size=0.01,
    tau=10000.0,
    kappa=2.0,
    normalizer_decay=0.99,
    hidden_sizes=(64, 64),
)

# SARSA defaults
DEFAULT_SARSA = dict(
    optimizer_name="autostep",
    initial_step_size=0.01,
    meta_step_size=0.01,
    tau=10000.0,
    kappa=2.0,
    normalizer_decay=0.99,
    epsilon=0.05,
    hidden_sizes=(64, 64),
)


def run_cartpole(
    seed: int,
    num_steps: int,
    factory: Any,
    config: dict[str, Any],
) -> tuple[float, list[float]]:
    raw_env = bsuite.load_from_id("cartpole/0")
    env = ContinuingWrapper(raw_env, mode="continuing")
    agent = factory(
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
        if float(new_timestep.discount) == 0.0:
            episode_returns.append(current_return)
            current_return = 0.0
        timestep = new_timestep
    if not episode_returns:
        return current_return, [current_return]
    half = max(1, len(episode_returns) // 2)
    return float(np.mean(episode_returns[-half:])), episode_returns


def run_catch(
    seed: int,
    num_steps: int,
    factory: Any,
    config: dict[str, Any],
) -> tuple[float, float, int]:
    """Run catch/0 trial.

    Returns (total_reward, total_regret, num_episodes).
    For catch optimal_return = 1, regret = 1 - reward.
    """
    raw_env = bsuite.load_from_id("catch/0")
    env = ContinuingWrapper(raw_env, mode="continuing")
    agent = factory(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        seed=seed,
        **config,
    )
    timestep = env.reset()
    total_reward = 0.0
    total_regret = 0.0
    n_episodes = 0
    current_return = 0.0
    for _ in range(num_steps):
        action = agent.select_action(timestep)
        new_timestep = env.step(action)
        agent.update(timestep, action, new_timestep)
        reward = float(new_timestep.reward) if new_timestep.reward is not None else 0.0
        current_return += reward
        total_reward += reward
        if float(new_timestep.discount) == 0.0:
            n_episodes += 1
            # Optimal catch return is 1.0; regret per episode = 1 - actual_return
            total_regret += 1.0 - current_return
            current_return = 0.0
        timestep = new_timestep
    return total_reward, total_regret, n_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--cartpole_steps", type=int, default=2000)
    parser.add_argument("--catch_steps", type=int, default=2700)  # ~300 episodes
    parser.add_argument(
        "--output", type=str, default="outputs/step4_ac_sweep/comparison_10seed.json"
    )
    args = parser.parse_args()

    configurations: dict[str, tuple[Any, dict[str, Any]]] = {
        "actor_critic_default": (actor_critic.default_agent, DEFAULT_AC),
        "actor_critic_tuned": (actor_critic.default_agent, TUNED_AC),
        "q_autostep": (autostep_dqn.default_agent, DEFAULT_QL),
        "sarsa": (sarsa_agent.default_agent, DEFAULT_SARSA),
    }

    cartpole_results: dict[str, list[dict[str, Any]]] = {}
    catch_results: dict[str, list[dict[str, Any]]] = {}

    t_start = time.time()
    for name, (factory, cfg) in configurations.items():
        print(f"\n=== {name} ===")
        cartpole_results[name] = []
        catch_results[name] = []
        for seed in range(args.num_seeds):
            t0 = time.time()
            mean_recent, ep_returns = run_cartpole(seed, args.cartpole_steps, factory, cfg)
            cp_t = time.time() - t0
            t0 = time.time()
            tot_r, tot_regret, n_ep = run_catch(seed, args.catch_steps, factory, cfg)
            ca_t = time.time() - t0
            cartpole_results[name].append(
                {"seed": seed, "mean_recent": mean_recent, "episode_returns": ep_returns}
            )
            catch_results[name].append(
                {
                    "seed": seed,
                    "total_reward": tot_r,
                    "total_regret": tot_regret,
                    "n_episodes": n_ep,
                }
            )
            print(
                f"  seed{seed}: cp_recent={mean_recent:6.2f} ({cp_t:.1f}s)  "
                f"catch_regret={tot_regret:6.2f} ep={n_ep} ({ca_t:.1f}s)  "
                f"total {time.time()-t_start:.0f}s"
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {"cartpole": cartpole_results, "catch": catch_results}, f, indent=2, default=str
        )

    # Summary
    print("\n\n=== SUMMARY ===")
    header = (
        f"{'config':30s} {'cartpole_mean':>15s} {'cartpole_std':>15s} "
        f"{'catch_regret_mean':>20s} {'catch_regret_std':>20s}"
    )
    print(header)
    for name in configurations:
        cps = [r["mean_recent"] for r in cartpole_results[name]]
        cas = [r["total_regret"] for r in catch_results[name]]
        print(
            f"{name:30s} {np.mean(cps):>15.2f} {np.std(cps):>15.2f} "
            f"{np.mean(cas):>20.2f} {np.std(cas):>20.2f}"
        )


if __name__ == "__main__":
    main()

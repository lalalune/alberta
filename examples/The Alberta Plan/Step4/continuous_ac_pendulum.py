"""Continuous-action actor-critic preview on Pendulum-v1.

Alberta Plan Step 4 (preview): on-policy continuous control with a linear
diagonal-Gaussian actor and a linear value critic, both equipped with
accumulating eligibility traces. Compares against a uniform-random baseline
across 5 seeds and reports mean episodic return per 1000-step window.

Usage:
    python "examples/The Alberta Plan/Step4/continuous_ac_pendulum.py" \
        --output-dir outputs/continuous_control_preview
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    ObGDBounding,
)


def _has_gymnasium() -> bool:
    try:
        import gymnasium  # noqa: F401

        return True
    except ImportError:
        return False


def _windowed_returns(
    episode_returns: list[float],
    episode_lengths: list[int],
    window: int = 1000,
) -> list[float]:
    """Average episodic return per ``window`` env steps (windowed by step index).

    Each completed episode contributes its return to the window in which its
    final step landed. Empty windows are filled with NaN.
    """
    if not episode_returns:
        return []
    total_steps = sum(episode_lengths)
    n_windows = max(1, (total_steps + window - 1) // window)
    sums = [0.0] * n_windows
    counts = [0] * n_windows
    cum = 0
    for ret, length in zip(episode_returns, episode_lengths, strict=True):
        cum += length
        idx = min((cum - 1) // window, n_windows - 1)
        sums[idx] += ret
        counts[idx] += 1
    out: list[float] = []
    for s, c in zip(sums, counts, strict=True):
        out.append(s / c if c > 0 else float("nan"))
    return out


def run_random_policy(
    seed: int, num_steps: int, action_low: float, action_high: float
) -> tuple[list[float], list[int]]:
    """Run a uniform-random behavior policy on Pendulum-v1 for ``num_steps``."""
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    ep_return = 0.0
    ep_length = 0
    for _ in range(num_steps):
        action = rng.uniform(action_low, action_high, size=(1,)).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)
        ep_length += 1
        if terminated or truncated:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            ep_return = 0.0
            ep_length = 0
            obs, _ = env.reset()
    if ep_length > 0:
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
    env.close()
    return episode_returns, episode_lengths


def run_continuous_ac(
    seed: int,
    num_steps: int,
    action_low: float,
    action_high: float,
) -> tuple[list[float], list[int]]:
    """Run the continuous-action actor-critic on Pendulum-v1 for ``num_steps``."""
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    obs_shape = env.observation_space.shape or (1,)
    feature_dim = int(np.prod(obs_shape))

    config = ContinuousActorCriticConfig(
        action_dim=1,
        gamma=0.99,
        actor_step_size=3e-5,
        critic_step_size=5e-3,
        actor_lamda=0.9,
        critic_lamda=0.9,
        log_sigma_init=0.0,
        log_sigma_min=-2.0,
        log_sigma_max=1.0,
        action_low=action_low,
        action_high=action_high,
    )
    agent = ContinuousActorCriticAgent(config, bounder=ObGDBounding(kappa=2.0))
    state = agent.init(feature_dim=feature_dim, key=jr.key(seed))

    raw_obs, _ = env.reset(seed=seed)
    obs = jnp.asarray(raw_obs, dtype=jnp.float32)
    state, action, _mean, _sigma = agent.start(state, obs)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    ep_return = 0.0
    ep_length = 0
    for _ in range(num_steps):
        env_action = np.asarray(action, dtype=np.float32).reshape(
            env.action_space.shape
        )
        raw_next_obs, reward, terminated, truncated, _ = env.step(env_action)
        next_obs = jnp.asarray(raw_next_obs, dtype=jnp.float32)
        ep_return += float(reward)
        ep_length += 1
        done = bool(terminated or truncated)
        # On time-limit truncation Pendulum-v1 resets without a true terminal
        # transition; we still bootstrap through the end of the episode by
        # passing terminated only on real termination (Pendulum has none).
        result = agent.update(
            state,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            observation=next_obs,
            terminated=jnp.asarray(bool(terminated)),
        )
        state = result.state
        action = result.action
        if done:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            ep_return = 0.0
            ep_length = 0
            raw_obs, _ = env.reset()
            obs = jnp.asarray(raw_obs, dtype=jnp.float32)
            state, action, _mean, _sigma = agent.start(state, obs)
    if ep_length > 0:
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
    env.close()
    return episode_returns, episode_lengths


def _summarise(
    label: str,
    seed_results: list[tuple[list[float], list[int]]],
    window: int = 1000,
) -> dict[str, object]:
    per_seed_windows = [
        _windowed_returns(rets, lens, window=window) for rets, lens in seed_results
    ]
    max_windows = max((len(w) for w in per_seed_windows), default=0)
    padded = []
    for w in per_seed_windows:
        if len(w) < max_windows:
            w = w + [float("nan")] * (max_windows - len(w))
        padded.append(w)
    arr = np.array(padded, dtype=np.float64)
    mean_curve = np.nanmean(arr, axis=0).tolist() if arr.size else []
    std_curve = np.nanstd(arr, axis=0).tolist() if arr.size else []
    final_window = mean_curve[-1] if mean_curve else float("nan")
    first_window = mean_curve[0] if mean_curve else float("nan")
    all_returns = [r for rets, _ in seed_results for r in rets]
    overall_mean = float(np.mean(all_returns)) if all_returns else float("nan")
    return {
        "label": label,
        "first_window_mean_return": float(first_window),
        "final_window_mean_return": float(final_window),
        "overall_mean_episodic_return": overall_mean,
        "mean_curve_per_window": mean_curve,
        "std_curve_per_window": std_curve,
        "n_seeds": len(seed_results),
        "n_windows": max_windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous AC on Pendulum-v1")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/continuous_control_preview",
        help="Directory for output JSON and summary files",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-steps", type=int, default=50_000)
    parser.add_argument("--window", type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _has_gymnasium():
        skip_summary = (
            "# Pendulum-v1 continuous AC preview\n\n"
            "Skipped: gymnasium is not installed in the environment. The class\n"
            "implementation and unit tests still run; install `gymnasium` to\n"
            "rerun this demo.\n"
        )
        (output_dir / "SUMMARY.md").write_text(skip_summary)
        print("gymnasium not available; skipping run.")
        return

    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    action_space = env.action_space
    action_low = float(action_space.low[0])  # type: ignore[attr-defined]
    action_high = float(action_space.high[0])  # type: ignore[attr-defined]
    env.close()

    print(
        f"Pendulum-v1: action bounds [{action_low}, {action_high}], "
        f"{args.num_steps} steps per seed across seeds {args.seeds}."
    )

    ac_runs: list[tuple[list[float], list[int]]] = []
    random_runs: list[tuple[list[float], list[int]]] = []
    for seed in args.seeds:
        t0 = time.time()
        rets_random, lens_random = run_random_policy(
            seed=seed,
            num_steps=args.num_steps,
            action_low=action_low,
            action_high=action_high,
        )
        random_runs.append((rets_random, lens_random))
        rets_ac, lens_ac = run_continuous_ac(
            seed=seed,
            num_steps=args.num_steps,
            action_low=action_low,
            action_high=action_high,
        )
        ac_runs.append((rets_ac, lens_ac))
        elapsed = time.time() - t0
        print(
            f"  seed={seed} done in {elapsed:.1f}s "
            f"(random eps={len(rets_random)}, ac eps={len(rets_ac)})"
        )

    summary_random = _summarise("random_uniform", random_runs, window=args.window)
    summary_ac = _summarise("continuous_actor_critic", ac_runs, window=args.window)
    output = {
        "env": "Pendulum-v1",
        "num_steps_per_seed": args.num_steps,
        "window": args.window,
        "seeds": list(args.seeds),
        "results": {
            "continuous_actor_critic": summary_ac,
            "random_uniform": summary_random,
        },
    }
    (output_dir / "pendulum_5seed_results.json").write_text(json.dumps(output, indent=2))

    # Markdown summary
    ac_first = summary_ac["first_window_mean_return"]
    ac_final = summary_ac["final_window_mean_return"]
    rand_first = summary_random["first_window_mean_return"]
    rand_final = summary_random["final_window_mean_return"]
    delta_first = ac_first - rand_first  # type: ignore[operator]
    delta_final = ac_final - rand_final  # type: ignore[operator]
    md_lines = [
        "# Pendulum-v1 continuous actor-critic preview",
        "",
        f"Seeds: {args.seeds}, steps per seed: {args.num_steps}, window: {args.window}.",
        "",
        "## Mean episodic return (per 1000-step window)",
        "",
        "| Agent | First window | Final window | Overall mean |",
        "|-------|--------------|--------------|--------------|",
        (
            "| Continuous AC | "
            f"{summary_ac['first_window_mean_return']:.2f} | "
            f"{summary_ac['final_window_mean_return']:.2f} | "
            f"{summary_ac['overall_mean_episodic_return']:.2f} |"
        ),
        (
            "| Random uniform | "
            f"{summary_random['first_window_mean_return']:.2f} | "
            f"{summary_random['final_window_mean_return']:.2f} | "
            f"{summary_random['overall_mean_episodic_return']:.2f} |"
        ),
        "",
        f"Delta first window (AC - random): {delta_first:.2f}",
        "",
        f"Delta final window (AC - random): {delta_final:.2f}",
        "",
        "Pendulum-v1 returns are negative (cost) and bounded above by 0.",
        "Higher (less negative) is better. The action space is "
        f"[{action_low}, {action_high}].",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {output_dir / 'pendulum_5seed_results.json'}")
    print(f"Wrote {output_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()

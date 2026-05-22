"""Step 7 planning benchmark on nonlinear feature observations.

This diagnostic is intentionally lightweight: the environment is the same
six-state continuing chain used by the tabular Step 7 benchmark, but the agent
only sees dense nonlinear Fourier features.  Dyna planning uses a learned
feature-space one-step model, so the evidence is beyond the one-state and
tabular planning checks while remaining fast enough for a local gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_STATES = 6
N_ACTIONS = 2
STATE_COORDS = np.linspace(-1.0, 1.0, N_STATES)


def _features(state: int) -> np.ndarray:
    coord = STATE_COORDS[state]
    return np.array(
        [
            coord,
            coord * coord,
            np.sin(np.pi * coord),
            np.cos(np.pi * coord),
            np.sin(2.0 * np.pi * coord),
            np.cos(2.0 * np.pi * coord),
            1.0,
        ],
        dtype=np.float64,
    )


def _model_features(observation: np.ndarray, action: int) -> np.ndarray:
    action_one_hot = np.zeros(N_ACTIONS, dtype=np.float64)
    action_one_hot[action] = 1.0
    return np.concatenate([observation, action_one_hot])


def _step_env(state: int, action: int) -> tuple[int, float]:
    if action == 0:
        next_state = max(0, state - 1)
        reward = 0.005 if state == 0 else 0.0
    else:
        next_state = min(N_STATES - 1, state + 1)
        reward = 1.0 if next_state == N_STATES - 1 else 0.0
    return next_state, reward


def _choose_action(q_values: np.ndarray, rng: np.random.Generator, epsilon: float) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    tie_noise = rng.gumbel(size=N_ACTIONS) * 1e-6
    return int(np.argmax(q_values + tie_noise))


def _run_seed(
    seed: int,
    *,
    steps: int,
    planning_steps: int,
    use_dyna: bool,
) -> dict[str, float | int | list[float]]:
    rng = np.random.default_rng(seed)
    feature_dim = len(_features(0))
    model_dim = feature_dim + N_ACTIONS
    q_weights = np.zeros((N_ACTIONS, feature_dim), dtype=np.float64)
    model_weights = np.zeros((model_dim, feature_dim + 1), dtype=np.float64)
    average_reward = 0.0
    q_step_size = 0.08
    average_reward_step_size = 0.01
    model_step_size = 0.20
    memory: list[tuple[np.ndarray, int, float]] = []

    state = int(rng.integers(N_STATES))
    observation = _features(state)
    action = _choose_action(q_weights @ observation, rng, epsilon=0.20)
    rewards: list[float] = []

    for step in range(steps):
        epsilon = max(0.02, 0.20 - 0.18 * step / steps)
        next_state, reward = _step_env(state, action)
        next_observation = _features(next_state)
        next_action = _choose_action(q_weights @ next_observation, rng, epsilon)
        td_error = (
            reward
            - average_reward
            + q_weights[next_action] @ next_observation
            - q_weights[action] @ observation
        )
        q_weights[action] += q_step_size * td_error * observation
        average_reward += average_reward_step_size * td_error
        rewards.append(float(reward))

        model_input = _model_features(observation, action)
        model_target = np.concatenate([[reward], next_observation])
        model_error = model_target - model_input @ model_weights
        model_weights += (
            model_step_size
            * np.outer(model_input, model_error)
            / (1.0 + model_input @ model_input)
        )
        priority = float(abs(model_error[0]) + np.linalg.norm(model_error[1:]))
        memory.append((observation.copy(), action, priority))
        memory = memory[-64:]

        if use_dyna and step >= 5:
            for _ in range(planning_steps):
                planned_observation, planned_action, _priority = max(
                    memory,
                    key=lambda item: item[2],
                )
                prediction = _model_features(planned_observation, planned_action) @ (
                    model_weights
                )
                planned_reward = float(prediction[0])
                planned_next_observation = prediction[1:]
                planned_next_action = _choose_action(
                    q_weights @ planned_next_observation,
                    rng,
                    epsilon=0.0,
                )
                planned_td_error = (
                    planned_reward
                    - average_reward
                    + q_weights[planned_next_action] @ planned_next_observation
                    - q_weights[planned_action] @ planned_observation
                )
                q_weights[planned_action] += (
                    q_step_size * planned_td_error * planned_observation
                )
                average_reward += average_reward_step_size * planned_td_error

        state = next_state
        observation = next_observation
        action = next_action

    return {
        "seed": seed,
        "final_window_reward": float(np.mean(rewards[-50:])),
        "cumulative_reward": float(np.sum(rewards)),
        "average_reward_estimate": float(average_reward),
        "rewards": rewards,
    }


def run_benchmark(
    *,
    seeds: int = 10,
    steps: int = 300,
    planning_steps: int = 8,
) -> dict[str, object]:
    """Run nonlinear-feature real-only vs. Dyna planning comparison."""
    seed_values = list(range(seeds))
    real_results = [
        _run_seed(seed, steps=steps, planning_steps=0, use_dyna=False)
        for seed in seed_values
    ]
    dyna_results = [
        _run_seed(seed, steps=steps, planning_steps=planning_steps, use_dyna=True)
        for seed in seed_values
    ]
    real_finals = np.array([result["final_window_reward"] for result in real_results])
    dyna_finals = np.array([result["final_window_reward"] for result in dyna_results])
    real_cumulative = np.array([result["cumulative_reward"] for result in real_results])
    dyna_cumulative = np.array([result["cumulative_reward"] for result in dyna_results])
    final_improvements = dyna_finals - real_finals
    cumulative_improvements = dyna_cumulative - real_cumulative
    passed = bool(
        np.mean(final_improvements) >= 0.08
        and np.sum(final_improvements > 0.0) >= 6
        and np.mean(cumulative_improvements) > 0.0
    )
    return {
        "schema": "alberta.step7.nonlinear_feature_planning.v1",
        "claim_scope": "nonlinear_feature_observation_dyna_planning",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "planning_steps": planning_steps,
            "feature_dim": len(_features(0)),
        },
        "aggregate": {
            "mean_real_only_final_window_reward": float(np.mean(real_finals)),
            "mean_dyna_final_window_reward": float(np.mean(dyna_finals)),
            "mean_final_window_improvement": float(np.mean(final_improvements)),
            "final_window_win_count": int(np.sum(final_improvements > 0.0)),
            "mean_real_only_cumulative_reward": float(np.mean(real_cumulative)),
            "mean_dyna_cumulative_reward": float(np.mean(dyna_cumulative)),
            "mean_cumulative_improvement": float(np.mean(cumulative_improvements)),
            "cumulative_win_count": int(np.sum(cumulative_improvements > 0.0)),
            "passed": passed,
        },
        "per_seed": [
            {
                "seed": seed,
                "real_only_final_window_reward": float(real_finals[index]),
                "dyna_final_window_reward": float(dyna_finals[index]),
                "final_window_improvement": float(final_improvements[index]),
                "real_only_cumulative_reward": float(real_cumulative[index]),
                "dyna_cumulative_reward": float(dyna_cumulative[index]),
                "cumulative_improvement": float(cumulative_improvements[index]),
            }
            for index, seed in enumerate(seed_values)
        ],
    }


def main() -> None:
    """Run benchmark and write the standard artifact."""
    result = run_benchmark()
    output_dir = Path("outputs/step7_nonlinear_feature_planning")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

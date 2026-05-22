"""Step 8 probes for behavior prediction, working memory, and dreaming.

The probes are deliberately small and deterministic enough to run locally:

* behavior prediction learns a stochastic behavior policy from online actions;
* working memory exposes a one-step delayed action target;
* guarded Dyna uses an action-conditioned world model to add imagined reward
  prediction updates after real updates.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.behavior_model import BehaviorModel, BehaviorModelConfig
from alberta_framework.core.reward_model import RLSRewardModel, RLSRewardModelConfig
from alberta_framework.core.working_memory import (
    WorkingMemoryConfig,
    WorkingMemoryFeaturizer,
    transform_working_memory_arrays,
)
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)


@dataclass(frozen=True)
class ProbeConfig:
    """Shared probe configuration."""

    seeds: int = 10
    steps: int = 600
    final_window: int = 100
    dreams_per_step: int = 1
    dream_warmup: int = 80
    dream_step_size: float = 0.006
    dream_error_threshold: float = 0.15
    dream_reward_error_threshold: float = 999.0
    dream_reward_error_decay: float = 0.95
    dream_decay: float = 0.25
    dream_loss_pressure_threshold: float = 0.0
    dream_loss_pressure_decay: float = 0.99
    dream_action_mode: str = "counterfactual"
    dream_target_mode: str = "reward_model"
    reward_model_mode: str = "rls"
    reward_model_rls_lambda: float = 0.995
    reward_model_rls_ridge: float = 10.0
    world_interactions: bool = True
    world_step_size: float = 0.06
    output_dir: str = "output/model_memory_dreaming_probe"


def _sigmoid(value: Array) -> Array:
    return jax.nn.sigmoid(value)


def _true_policy_probability(x_value: Array) -> Array:
    return _sigmoid(4.0 * x_value)


def _env_step(x_value: Array, action: Array) -> tuple[Array, Array]:
    action_f = action.astype(jnp.float32)
    signed = 2.0 * action_f - 1.0
    next_x = jnp.clip(0.82 * x_value + 0.16 * signed, -1.5, 1.5)
    reward = 0.35 * next_x + 0.22 * action_f + 0.12 * x_value * action_f
    return next_x, reward


def _reward_features(x_value: Array, action: Array) -> Array:
    action_f = action.astype(jnp.float32)
    return jnp.array(
        [1.0, x_value, action_f, x_value * action_f],
        dtype=jnp.float32,
    )


def _linear_update(weights: Array, features: Array, target: Array, step_size: float) -> Array:
    prediction = jnp.dot(weights, features)
    error = target - prediction
    return weights + step_size * error * features


def _reward_mse(weights: Array) -> float:
    xs = jnp.linspace(-1.0, 1.0, 101)
    errors = []
    for action in (jnp.array(0, dtype=jnp.int32), jnp.array(1, dtype=jnp.int32)):
        for x_value in xs:
            _, reward = _env_step(x_value, action)
            pred = jnp.dot(weights, _reward_features(x_value, action))
            errors.append((pred - reward) ** 2)
    return float(jnp.mean(jnp.asarray(errors)))


def run_behavior_probe(config: ProbeConfig) -> dict[str, float]:
    """Run online behavior-policy learning against a stochastic policy."""
    final_nll = []
    final_accuracy = []
    uniform_nll = float(np.log(2.0))
    for seed in range(config.seeds):
        key = jr.key(seed)
        model = BehaviorModel(
            BehaviorModelConfig(n_actions=2, step_size=0.08, diagnostic_decay=0.95)
        )
        state = model.init(feature_dim=1, key=key)
        nlls = []
        correct = []
        x_value = jnp.array(0.0, dtype=jnp.float32)
        for step in range(config.steps):
            key, obs_key, act_key = jr.split(key, 3)
            x_value = 0.93 * x_value + 0.2 * jr.normal(obs_key, ())
            probability = _true_policy_probability(x_value)
            action = jr.bernoulli(act_key, probability).astype(jnp.int32)
            result = model.update(state, jnp.reshape(x_value, (1,)), action)
            state = result.state
            if step >= config.steps - config.final_window:
                nlls.append(float(result.loss))
                correct.append(float(result.correct))
        final_nll.append(float(np.mean(nlls)))
        final_accuracy.append(float(np.mean(correct)))
    return {
        "uniform_nll": uniform_nll,
        "final_nll_mean": float(np.mean(final_nll)),
        "final_nll_se": (
            0.0
            if config.seeds < 2
            else float(np.std(final_nll, ddof=1) / np.sqrt(config.seeds))
        ),
        "nll_improvement": float(uniform_nll - np.mean(final_nll)),
        "final_accuracy_mean": float(np.mean(final_accuracy)),
    }


def run_working_memory_probe(config: ProbeConfig) -> dict[str, float]:
    """Run a one-step delayed-action target probe."""
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=2,
            reward_dim=0,
            observation_decay_rates=(),
            action_decay_rates=(0.0,),
            reward_decay_rates=(),
            include_current_observation=False,
            include_current_action=False,
            include_current_reward=False,
        )
    )
    memory_mses = []
    raw_mses = []
    for seed in range(config.seeds):
        key = jr.key(10_000 + seed)
        action_ids = jr.bernoulli(key, 0.5, (config.steps,)).astype(jnp.int32)
        actions = jax.nn.one_hot(action_ids, 2)
        observations = jnp.zeros((config.steps, 1), dtype=jnp.float32)
        rewards = jnp.zeros((config.steps, 0), dtype=jnp.float32)
        _, features = transform_working_memory_arrays(
            memory,
            observations,
            actions,
            rewards,
        )
        target = jnp.concatenate(
            [jnp.asarray([0.0], dtype=jnp.float32), actions[:-1, 1]],
            axis=0,
        )
        memory_prediction = features[:, 1]
        raw_prediction = jnp.full_like(target, 0.5)
        memory_mses.append(float(jnp.mean((memory_prediction - target) ** 2)))
        raw_mses.append(float(jnp.mean((raw_prediction - target) ** 2)))
    return {
        "raw_mse_mean": float(np.mean(raw_mses)),
        "memory_mse_mean": float(np.mean(memory_mses)),
        "mse_improvement": float(np.mean(raw_mses) - np.mean(memory_mses)),
        "wins": float(sum(m < r for m, r in zip(memory_mses, raw_mses))),
    }


def run_dreaming_probe(config: ProbeConfig) -> dict[str, float]:
    """Run a guarded Dyna reward-prediction probe on a simple controlled system."""
    real_mses = []
    dyna_mses = []
    accepted_rates = []
    model_errors = []
    reward_error_emas = []
    for seed in range(config.seeds):
        key = jr.key(20_000 + seed)
        world = ActionConditionedWorldModel(
            ActionConditionedWorldModelConfig(
                observation_dim=1,
                n_actions=2,
                hidden_sizes=(),
                step_size=config.world_step_size,
                sparsity=0.0,
                use_layer_norm=False,
                gamma=0.99,
                max_delta_scale=3.0,
                include_action_interactions=config.world_interactions,
            )
        )
        behavior = BehaviorModel(BehaviorModelConfig(n_actions=2, step_size=0.08))
        world_state = world.init(jr.fold_in(key, 1))
        behavior_state = behavior.init(feature_dim=1, key=jr.fold_in(key, 2))
        reward_model = RLSRewardModel(
            RLSRewardModelConfig(
                feature_dim=4,
                forgetting=config.reward_model_rls_lambda,
                ridge=config.reward_model_rls_ridge,
            )
        )
        rls_reward_state = reward_model.init()
        real_weights = jnp.zeros((4,), dtype=jnp.float32)
        dyna_weights = jnp.zeros((4,), dtype=jnp.float32)
        reward_model_weights = jnp.zeros((4,), dtype=jnp.float32)
        dyna_loss_pressure = 0.0
        reward_abs_error_ema = jnp.full((2,), jnp.inf, dtype=jnp.float32)
        anchors: list[float] = []
        accepted = 0
        proposed = 0
        x_value = jnp.array(0.0, dtype=jnp.float32)
        for step in range(config.steps):
            key, action_key, dream_key = jr.split(key, 3)
            probability = _true_policy_probability(x_value)
            action = jr.bernoulli(action_key, probability).astype(jnp.int32)
            next_x, reward = _env_step(x_value, action)
            features = _reward_features(x_value, action)
            real_weights = _linear_update(real_weights, features, reward, 0.025)
            if config.reward_model_mode == "lms":
                reward_model_weights = _linear_update(
                    reward_model_weights,
                    features,
                    reward,
                    config.world_step_size,
                )
            elif config.reward_model_mode == "rls":
                rls_reward_state = reward_model.update(
                    rls_reward_state,
                    features,
                    reward,
                ).state
            else:
                raise ValueError("reward_model_mode must be 'lms' or 'rls'")
            dyna_prediction = jnp.dot(dyna_weights, features)
            dyna_error = reward - dyna_prediction
            dyna_loss_pressure = (
                config.dream_loss_pressure_decay * dyna_loss_pressure
                + (1.0 - config.dream_loss_pressure_decay) * float(dyna_error**2)
            )
            dyna_weights = _linear_update(dyna_weights, features, reward, 0.025)
            world_result = world.update(
                world_state,
                jnp.reshape(x_value, (1,)),
                action,
                reward,
                jnp.array(0.99, dtype=jnp.float32),
                jnp.reshape(next_x, (1,)),
            )
            world_state = world_result.state
            action_index = int(action)
            previous_reward_error = reward_abs_error_ema[action_index]
            next_reward_error = jnp.abs(world_result.reward_error)
            reward_abs_error_ema = reward_abs_error_ema.at[action_index].set(
                jnp.where(
                    jnp.isfinite(previous_reward_error),
                    config.dream_reward_error_decay * previous_reward_error
                    + (1.0 - config.dream_reward_error_decay) * next_reward_error,
                    next_reward_error,
                )
            )
            behavior_state = behavior.update(
                behavior_state,
                jnp.reshape(x_value, (1,)),
                action,
            ).state
            anchors.append(float(x_value))
            can_dream = (
                step >= config.dream_warmup
                and float(world_state.model_error_ema) < config.dream_error_threshold
                and dyna_loss_pressure > config.dream_loss_pressure_threshold
            )
            if can_dream and anchors:
                for dream_idx in range(config.dreams_per_step):
                    proposed += 1
                    dream_key, anchor_key = jr.split(dream_key)
                    anchor_idx = int(
                        jr.randint(anchor_key, (), 0, len(anchors), dtype=jnp.int32)
                    )
                    anchor = jnp.array(anchors[anchor_idx], dtype=jnp.float32)
                    if config.dream_action_mode == "behavior":
                        sample = behavior.sample_action(
                            behavior_state,
                            jnp.reshape(anchor, (1,)),
                        )
                        behavior_state = sample.state
                        dream_action = sample.action
                    elif config.dream_action_mode == "counterfactual":
                        dream_action = jnp.array(dream_idx % 2, dtype=jnp.int32)
                    else:
                        raise ValueError(
                            "dream_action_mode must be 'behavior' or 'counterfactual'"
                        )
                    dream = world.predict(
                        world_state,
                        jnp.reshape(anchor, (1,)),
                        dream_action,
                    )
                    dream_reward_error = reward_abs_error_ema[int(dream_action)]
                    reward_head_trusted = (
                        bool(jnp.isfinite(dream_reward_error))
                        and float(dream_reward_error)
                        < config.dream_reward_error_threshold
                    )
                    if (
                        bool(jnp.all(jnp.isfinite(dream.next_observation)))
                        and reward_head_trusted
                    ):
                        accepted += 1
                        dream_features = _reward_features(anchor, dream_action)
                        if config.dream_target_mode == "model":
                            dream_reward = dream.reward
                        elif config.dream_target_mode == "reward_model":
                            if config.reward_model_mode == "rls":
                                dream_reward = reward_model.predict(
                                    rls_reward_state,
                                    dream_features,
                                )
                            else:
                                dream_reward = jnp.dot(
                                    reward_model_weights,
                                    dream_features,
                                )
                        elif config.dream_target_mode == "oracle":
                            _, dream_reward = _env_step(anchor, dream_action)
                        else:
                            raise ValueError(
                                "dream_target_mode must be 'model', "
                                "'reward_model', or 'oracle'"
                            )
                        model_confidence = max(
                            0.0,
                            1.0
                            - float(world_state.model_error_ema)
                            / config.dream_error_threshold,
                        )
                        dyna_weights = _linear_update(
                            dyna_weights,
                            dream_features,
                            dream_reward,
                            (config.dream_step_size * model_confidence)
                            / (1.0 + config.dream_decay * dream_idx),
                        )
            x_value = next_x
        real_mses.append(_reward_mse(real_weights))
        dyna_mses.append(_reward_mse(dyna_weights))
        accepted_rates.append(accepted / max(proposed, 1))
        model_errors.append(float(world_state.model_error_ema))
        finite_reward_errors = reward_abs_error_ema[jnp.isfinite(reward_abs_error_ema)]
        reward_error_emas.append(
            float(jnp.mean(finite_reward_errors))
            if finite_reward_errors.size
            else float("inf")
        )
    return {
        "real_mse_mean": float(np.mean(real_mses)),
        "dyna_mse_mean": float(np.mean(dyna_mses)),
        "mse_improvement": float(np.mean(real_mses) - np.mean(dyna_mses)),
        "dyna_wins": float(sum(d < r for d, r in zip(dyna_mses, real_mses))),
        "accepted_rate_mean": float(np.mean(accepted_rates)),
        "model_error_ema_mean": float(np.mean(model_errors)),
        "reward_error_ema_mean": float(np.mean(reward_error_emas)),
    }


def write_outputs(
    config: ProbeConfig,
    behavior: dict[str, float],
    memory: dict[str, float],
    dreaming: dict[str, float],
) -> None:
    """Write JSON, CSV, and Markdown summaries."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "behavior": behavior,
        "working_memory": memory,
        "dreaming": dreaming,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["probe", "metric", "value"])
        for probe, metrics in payload.items():
            if probe == "config":
                continue
            for metric, value in metrics.items():
                writer.writerow([probe, metric, value])
    lines = [
        "# Model, Memory, and Dreaming Probe",
        "",
        f"Seeds: `{config.seeds}`; steps: `{config.steps}`; "
        f"dreams per step: `{config.dreams_per_step}`.",
        f"Dream step size: `{config.dream_step_size}`; "
        f"error threshold: `{config.dream_error_threshold}`.",
        f"Reward-error threshold: `{config.dream_reward_error_threshold}`.",
        f"Loss-pressure threshold: `{config.dream_loss_pressure_threshold}`.",
        "",
        "## Behavior Prediction",
        "",
        f"- Final NLL: `{behavior['final_nll_mean']:.6f}` "
        f"+/- `{behavior['final_nll_se']:.6f}`.",
        f"- Uniform NLL improvement: `{behavior['nll_improvement']:.6f}`.",
        f"- Final accuracy: `{behavior['final_accuracy_mean']:.6f}`.",
        "",
        "## Working Memory",
        "",
        f"- Raw delayed-action MSE: `{memory['raw_mse_mean']:.6f}`.",
        f"- Memory delayed-action MSE: `{memory['memory_mse_mean']:.6f}`.",
        f"- Improvement: `{memory['mse_improvement']:.6f}`.",
        f"- Wins: `{memory['wins']:.0f}/{config.seeds}`.",
        "",
        "## Guarded Dreaming",
        "",
        f"- Real-only reward MSE: `{dreaming['real_mse_mean']:.6f}`.",
        f"- Real + dream reward MSE: `{dreaming['dyna_mse_mean']:.6f}`.",
        f"- Improvement: `{dreaming['mse_improvement']:.6f}`.",
        f"- Dyna wins: `{dreaming['dyna_wins']:.0f}/{config.seeds}`.",
        f"- Dream acceptance rate: `{dreaming['accepted_rate_mean']:.6f}`.",
        f"- Final model-error EMA: `{dreaming['model_error_ema_mean']:.6f}`.",
        f"- Final reward-error EMA: `{dreaming['reward_error_ema_mean']:.6f}`.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run all probes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--final-window", type=int, default=100)
    parser.add_argument("--dreams-per-step", type=int, default=1)
    parser.add_argument("--dream-warmup", type=int, default=80)
    parser.add_argument("--dream-step-size", type=float, default=0.006)
    parser.add_argument("--dream-error-threshold", type=float, default=0.15)
    parser.add_argument("--dream-reward-error-threshold", type=float, default=999.0)
    parser.add_argument("--dream-reward-error-decay", type=float, default=0.95)
    parser.add_argument("--dream-decay", type=float, default=0.25)
    parser.add_argument("--dream-loss-pressure-threshold", type=float, default=0.0)
    parser.add_argument("--dream-loss-pressure-decay", type=float, default=0.99)
    parser.add_argument(
        "--no-world-interactions",
        action="store_true",
        help="Disable observation-by-action interaction inputs in the world model.",
    )
    parser.add_argument("--world-step-size", type=float, default=0.06)
    parser.add_argument(
        "--dream-action-mode",
        choices=("behavior", "counterfactual"),
        default="counterfactual",
    )
    parser.add_argument(
        "--dream-target-mode",
        choices=("model", "reward_model", "oracle"),
        default="reward_model",
        help=(
            "Use multi-head world-model rewards, a separate reward model, "
            "or oracle probe rewards for imagined updates."
        ),
    )
    parser.add_argument("--reward-model-mode", choices=("lms", "rls"), default="rls")
    parser.add_argument("--reward-model-rls-lambda", type=float, default=0.995)
    parser.add_argument("--reward-model-rls-ridge", type=float, default=10.0)
    parser.add_argument("--output-dir", type=str, default="output/model_memory_dreaming_probe")
    args = parser.parse_args()
    config = ProbeConfig(
        seeds=args.seeds,
        steps=args.steps,
        final_window=args.final_window,
        dreams_per_step=args.dreams_per_step,
        dream_warmup=args.dream_warmup,
        dream_step_size=args.dream_step_size,
        dream_error_threshold=args.dream_error_threshold,
        dream_reward_error_threshold=args.dream_reward_error_threshold,
        dream_reward_error_decay=args.dream_reward_error_decay,
        dream_decay=args.dream_decay,
        dream_loss_pressure_threshold=args.dream_loss_pressure_threshold,
        dream_loss_pressure_decay=args.dream_loss_pressure_decay,
        dream_action_mode=args.dream_action_mode,
        dream_target_mode=args.dream_target_mode,
        reward_model_mode=args.reward_model_mode,
        reward_model_rls_lambda=args.reward_model_rls_lambda,
        reward_model_rls_ridge=args.reward_model_rls_ridge,
        world_interactions=not args.no_world_interactions,
        world_step_size=args.world_step_size,
        output_dir=args.output_dir,
    )
    behavior = run_behavior_probe(config)
    memory = run_working_memory_probe(config)
    dreaming = run_dreaming_probe(config)
    write_outputs(config, behavior, memory, dreaming)
    print(
        json.dumps(
            {"behavior": behavior, "working_memory": memory, "dreaming": dreaming},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""DoD-6: recurrent state on POMDP-RandomWalk.

This sweep replaces the earlier partial Pavlovian masking probe with the
requested POMDP-RandomWalk ablation:

- raw: masked observation only.
- trace_only: masked observation plus multi-timescale observation traces.
- gvf_feedback: masked observation plus previous auxiliary GVF predictions.
- trace_plus_gvf: both recurrent traces and GVF feedback.

The stream is a RandomWalk-style drifting linear target over an AR(1) latent
observation. A periodic partial-observation mask hides channels 2..5 on two
out of every three steps. The downstream target depends on the full latent
observation, so the masked problem is genuinely partially observable while
remaining recoverable from recurrent state.

Output: ``output/step3_dod6/{results.csv,summary.json}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework import (
    LMS,
    DemonType,
    GVFSpec,
    HistoryFeatureExtractor,
    HordeLearner,
    LinearLearner,
    MaskMode,
    PartialObservationWrapper,
    create_horde_spec,
)
from alberta_framework.core.types import TimeStep

OUTPUT_DIR = Path("output/step3_dod6")
FEATURE_DIM = 6
HIDDEN_CHANNELS = (2, 3, 4, 5)
DECAY_RATES = (0.5, 0.75, 0.9, 0.97)
N_SEEDS = 10
N_STEPS = 4000
FINAL_WINDOW = 1000
TARGET_ALPHA = 0.03
GVF_STEP_SIZE = 0.02


@dataclass(frozen=True)
class POMDPRandomWalkState:
    """State for the local AR POMDP-RandomWalk stream."""

    key: Array
    latent_observation: Array
    true_weights: Array


class AutoregressiveRandomWalkStream:
    """RandomWalk target stream with temporally correlated latent observations."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        obs_ar: float = 0.96,
        obs_noise_std: float = 0.20,
        drift_rate: float = 0.001,
        target_noise_std: float = 0.02,
    ) -> None:
        self._feature_dim = feature_dim
        self._obs_ar = obs_ar
        self._obs_noise_std = obs_noise_std
        self._drift_rate = drift_rate
        self._target_noise_std = target_noise_std

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def init(self, key: Array) -> POMDPRandomWalkState:
        key, k_obs, k_w = jr.split(key, 3)
        return POMDPRandomWalkState(
            key=key,
            latent_observation=jr.normal(k_obs, (self._feature_dim,), dtype=jnp.float32),
            true_weights=jr.normal(k_w, (self._feature_dim,), dtype=jnp.float32),
        )

    def step(
        self, state: POMDPRandomWalkState, idx: Array
    ) -> tuple[TimeStep, POMDPRandomWalkState]:
        del idx
        key, k_obs, k_drift, k_noise = jr.split(state.key, 4)
        innovation = self._obs_noise_std * jr.normal(
            k_obs, (self._feature_dim,), dtype=jnp.float32
        )
        latent = self._obs_ar * state.latent_observation + innovation
        drift = self._drift_rate * jr.normal(
            k_drift, (self._feature_dim,), dtype=jnp.float32
        )
        weights = state.true_weights + drift
        noise = self._target_noise_std * jr.normal(k_noise, (), dtype=jnp.float32)
        target = jnp.dot(weights, latent) + noise
        ts = TimeStep(  # type: ignore[call-arg]
            observation=latent, target=jnp.atleast_1d(target)
        )
        return ts, POMDPRandomWalkState(
            key=key, latent_observation=latent, true_weights=weights
        )


def _build_pomdp() -> PartialObservationWrapper[POMDPRandomWalkState]:
    inner = AutoregressiveRandomWalkStream()
    visible = jnp.ones(FEATURE_DIM, dtype=jnp.bool_)
    hidden = visible.at[jnp.asarray(HIDDEN_CHANNELS)].set(False)
    schedule = (visible, hidden, hidden)
    return PartialObservationWrapper(
        inner, mode=MaskMode.PERIODIC, schedule=schedule, sentinel=0.0
    )


def collect_pomdp_random_walk(seed: int, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect full latent observations, masked observations, and targets."""
    pomdp = _build_pomdp()
    state = pomdp.init(jr.key(seed))
    latent_rows: list[np.ndarray] = []
    masked_rows: list[np.ndarray] = []
    targets: list[float] = []

    for i in range(n_steps):
        ts, state = pomdp.step(state, jnp.array(i, dtype=jnp.int32))
        latent_rows.append(np.asarray(state.inner_state.latent_observation))
        masked_rows.append(np.asarray(ts.observation))
        targets.append(float(jnp.squeeze(ts.target)))

    return (
        np.asarray(latent_rows, dtype=np.float32),
        np.asarray(masked_rows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def _make_history_features(masked_obs: np.ndarray) -> np.ndarray:
    extractor = HistoryFeatureExtractor(
        raw_dim=masked_obs.shape[1], decay_rates=DECAY_RATES, include_raw=True
    )
    state = extractor.init()
    rows: list[np.ndarray] = []
    for obs in masked_obs:
        aug, state = extractor.step(state, jnp.asarray(obs))
        rows.append(np.asarray(aug))
    return np.asarray(rows, dtype=np.float32)


def _make_horde(feature_dim: int) -> HordeLearner:
    demons = [
        GVFSpec(  # type: ignore[call-arg]
            name=f"hidden_channel_{channel}",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=i,
        )
        for i, channel in enumerate(HIDDEN_CHANNELS)
    ]
    return HordeLearner(
        horde_spec=create_horde_spec(demons),
        hidden_sizes=(),
        step_size=GVF_STEP_SIZE,
        use_layer_norm=False,
    )


def _final_window_mse(sq_errors: list[float], final_window: int) -> float:
    return float(np.mean(sq_errors[-final_window:]))


def run_condition(
    *,
    seed: int,
    latent_obs: np.ndarray,
    masked_obs: np.ndarray,
    targets: np.ndarray,
    use_traces: bool,
    use_gvf_feedback: bool,
    final_window: int,
) -> float:
    base_features = _make_history_features(masked_obs) if use_traces else masked_obs
    feedback_dim = len(HIDDEN_CHANNELS) if use_gvf_feedback else 0
    learner = LinearLearner(optimizer=LMS(step_size=TARGET_ALPHA))
    learner_state = learner.init(base_features.shape[1] + feedback_dim)

    horde = _make_horde(base_features.shape[1]) if use_gvf_feedback else None
    horde_state = (
        horde.init(base_features.shape[1], jr.key(100_000 + seed))
        if horde is not None
        else None
    )
    feedback = jnp.zeros(feedback_dim, dtype=jnp.float32)
    sq_errors: list[float] = []

    for t in range(len(targets)):
        base = jnp.asarray(base_features[t])
        features = jnp.concatenate([base, feedback]) if use_gvf_feedback else base
        result = learner.update(
            learner_state,
            features,
            jnp.atleast_1d(jnp.float32(targets[t])),
        )
        learner_state = result.state
        sq_errors.append(float(jnp.squeeze(result.error) ** 2))

        if horde is not None and horde_state is not None:
            next_base = jnp.asarray(base_features[min(t + 1, len(targets) - 1)])
            cumulants = jnp.asarray(latent_obs[t, HIDDEN_CHANNELS])
            horde_result = horde.update(horde_state, base, cumulants, next_base)
            horde_state = horde_result.state
            feedback = horde.predict(horde_state, next_base)

    return _final_window_mse(sq_errors, final_window)


def run_seed(seed: int, n_steps: int, final_window: int) -> dict[str, float]:
    latent_obs, masked_obs, targets = collect_pomdp_random_walk(seed, n_steps)
    raw = run_condition(
        seed=seed,
        latent_obs=latent_obs,
        masked_obs=masked_obs,
        targets=targets,
        use_traces=False,
        use_gvf_feedback=False,
        final_window=final_window,
    )
    trace_only = run_condition(
        seed=seed,
        latent_obs=latent_obs,
        masked_obs=masked_obs,
        targets=targets,
        use_traces=True,
        use_gvf_feedback=False,
        final_window=final_window,
    )
    gvf_feedback = run_condition(
        seed=seed,
        latent_obs=latent_obs,
        masked_obs=masked_obs,
        targets=targets,
        use_traces=False,
        use_gvf_feedback=True,
        final_window=final_window,
    )
    trace_plus_gvf = run_condition(
        seed=seed,
        latent_obs=latent_obs,
        masked_obs=masked_obs,
        targets=targets,
        use_traces=True,
        use_gvf_feedback=True,
        final_window=final_window,
    )
    return {
        "seed": float(seed),
        "raw_mse": raw,
        "trace_only_mse": trace_only,
        "gvf_feedback_mse": gvf_feedback,
        "trace_plus_gvf_mse": trace_plus_gvf,
    }


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    raw = np.asarray([r["raw_mse"] for r in rows])
    summary: dict[str, float] = {"n_seeds": float(len(rows))}
    for condition in ("raw", "trace_only", "gvf_feedback", "trace_plus_gvf"):
        values = np.asarray([r[f"{condition}_mse"] for r in rows])
        summary[f"{condition}_mse_mean"] = float(np.mean(values))
        summary[f"{condition}_mse_std"] = float(np.std(values))
        if condition != "raw":
            diff = raw - values
            summary[f"{condition}_paired_diff_mean"] = float(np.mean(diff))
            summary[f"{condition}_better_seeds"] = float(np.sum(values < raw))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--final-window", type=int, default=FINAL_WINDOW)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows: list[dict[str, float]] = []

    for seed in range(args.seeds):
        row = run_seed(seed, args.steps, args.final_window)
        rows.append(row)
        print(
            f"seed={seed:>2}  raw={row['raw_mse']:.4f}  "
            f"trace={row['trace_only_mse']:.4f}  "
            f"gvf={row['gvf_feedback_mse']:.4f}  "
            f"both={row['trace_plus_gvf_mse']:.4f}"
        )

    csv_path = args.output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    t_total = time.perf_counter() - t0
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "settings": {
                    "n_steps": args.steps,
                    "final_window": args.final_window,
                    "hidden_channels": HIDDEN_CHANNELS,
                    "trace_decay_rates": DECAY_RATES,
                    "target_alpha": TARGET_ALPHA,
                    "gvf_step_size": GVF_STEP_SIZE,
                },
                "total_seconds": t_total,
            },
            f,
            indent=2,
        )

    print("\n=== DoD-6 POMDP-RandomWalk summary ===")
    for condition in ("raw", "trace_only", "gvf_feedback", "trace_plus_gvf"):
        print(
            f"  {condition:14s} MSE: "
            f"{summary[f'{condition}_mse_mean']:.4f} ± "
            f"{summary[f'{condition}_mse_std']:.4f}"
        )
    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Wrote {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

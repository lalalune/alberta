"""Pavlovian Horde: multi-horizon US anticipation in classical conditioning.

Demonstrates Step 3 of the Alberta Plan on the canonical Pavlovian
testbed: a Conditioned Stimulus (CS) precedes an Unconditioned Stimulus
(US) by a fixed delay, and a Horde of GVF prediction demons with
``gamma in {0, 0.5, 0.9, 0.99}`` learns the multi-horizon anticipation
of the US given the CS.

The ``gamma=0`` demon predicts the *next-step* US indicator; the
``gamma>0`` demons accumulate discounted future US activity, so their
predictions rise gradually from CS onset to US fire and persist longer
afterward — the classical "anticipation curve".

Usage:
    python "examples/The Alberta Plan/Step3/pavlovian_horde.py" --output-dir output/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

from alberta_framework import (
    DemonType,
    EMANormalizer,
    GVFSpec,
    HordeLearner,
    ObGDBounding,
    Timer,
    acquisition_scenario,
    create_horde_spec,
    run_horde_learning_loop,
)


def _build_horde_spec(gammas: tuple[float, ...]) -> tuple[GVFSpec, ...]:
    """Build one prediction demon per gamma, all sharing cumulant index 0."""
    demons = []
    for gamma in gammas:
        demons.append(
            GVFSpec(
                name=f"gamma_{gamma}",
                demon_type=DemonType.PREDICTION,
                gamma=gamma,
                lamda=0.0,
                cumulant_index=0,
            )
        )
    return tuple(demons)


def _collect_stream(
    stream, n_steps: int, key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Roll the stream out under ``jax.lax.scan`` and stack (obs, target)."""
    state = stream.init(key)

    def step_fn(carry, idx):
        ts, new_state = stream.step(carry, idx)
        return new_state, (ts.observation, ts.target)

    _, (obs, tgt) = jax.lax.scan(step_fn, state, jnp.arange(n_steps))
    return obs, tgt


def _running_mean(x: jnp.ndarray, window: int) -> jnp.ndarray:
    """Simple causal moving average (used for the loss curves)."""
    if window <= 1:
        return x
    pad = jnp.zeros((window - 1,), dtype=x.dtype)
    padded = jnp.concatenate([pad, x])
    kernel = jnp.ones((window,), dtype=x.dtype) / window
    return jnp.convolve(padded, kernel, mode="valid")


def main(
    output_dir: Path,
    num_steps: int = 5000,
    seed: int = 42,
    gammas: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99),
    cs_us_delay: int = 5,
    iti_min: int = 5,
    iti_max: int = 20,
    do_plot: bool = True,
) -> None:
    """Train a Horde of multi-horizon US predictors on the acquisition stream."""
    output_dir.mkdir(parents=True, exist_ok=True)

    key = jr.key(seed)
    k_stream, k_learner = jr.split(key)

    # Stream: single-CS acquisition.
    stream = acquisition_scenario(
        n_steps=num_steps,
        cs_us_delay=cs_us_delay,
        cs_duration=1,
        iti_min=iti_min,
        iti_max=iti_max,
        noise_std=0.05,
        distractor_prob=0.0,
    )
    feature_dim = stream.feature_dim

    # Horde with one demon per gamma, all watching the US indicator.
    demons = _build_horde_spec(gammas)
    horde_spec = create_horde_spec(demons)
    horde = HordeLearner(
        horde_spec=horde_spec,
        hidden_sizes=(64, 32),
        step_size=0.05,
        sparsity=0.9,
        normalizer=EMANormalizer(decay=0.99),
        bounder=ObGDBounding(kappa=2.0),
    )
    state = horde.init(feature_dim, k_learner)

    # Roll out the Pavlovian stream once.
    obs, tgt = _collect_stream(stream, num_steps, k_stream)
    # Cumulants are the US indicator, broadcast across all demons.
    n_demons = len(demons)
    cumulants = jnp.broadcast_to(tgt, (num_steps, n_demons))
    # Standard TD setup: next_obs is the next time step's observation. The
    # final step is paired with itself (degenerate), which has negligible
    # effect over thousands of steps.
    next_obs = jnp.concatenate([obs[1:], obs[-1:]], axis=0)

    # Per-step Horde predictions for plotting the anticipation curve. We
    # use ``jax.vmap`` over the rollout instead of running the learner
    # again — predictions are cheap relative to learning.
    with Timer("Horde learning loop"):
        result = run_horde_learning_loop(horde, state, obs, cumulants, next_obs)

    # Now re-run the predictions using the *final* state to see what the
    # learner has internalised. (For a learning curve over time use
    # ``result.per_demon_metrics``.)
    final_state = result.state
    final_predict = jax.vmap(lambda o: horde.predict(final_state, o))
    final_predictions = final_predict(obs)  # (num_steps, n_demons)

    # Print a brief summary per demon.
    print("\nPer-demon final-100-step squared error:")
    for i, demon in enumerate(demons):
        mean_se = float(jnp.nanmean(result.per_demon_metrics[-100:, i, 0]))
        max_pred = float(jnp.max(final_predictions[:, i]))
        mean_pred = float(jnp.mean(final_predictions[:, i]))
        print(
            f"  {demon.name:>10s} (gamma={demon.gamma:.2f}): "
            f"SE={mean_se:.4f}, "
            f"max_pred={max_pred:.3f}, "
            f"mean_pred={mean_pred:.3f}"
        )

    # Persist the full configuration so the run is reproducible.
    config = {
        "horde": horde.to_config(),
        "stream": {
            "type": "acquisition_scenario",
            "num_steps": num_steps,
            "cs_us_delay": cs_us_delay,
            "iti_min": iti_min,
            "iti_max": iti_max,
        },
        "seed": seed,
        "gammas": list(gammas),
    }
    config_path = output_dir / "pavlovian_horde_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    if not do_plot:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    # Plot: per-horizon prediction curves over time + raw stream signal.
    # We zoom on the last 200 steps so the trial structure is visible.
    window_start = max(0, num_steps - 200)
    t = jnp.arange(window_start, num_steps)
    cs_signal = obs[window_start:, 0]
    us_signal = tgt[window_start:, 0]

    fig, (ax_signal, ax_pred, ax_loss) = plt.subplots(
        3, 1, figsize=(10, 9), sharex=False
    )
    ax_signal.plot(t, cs_signal, label="CS", color="tab:blue")
    ax_signal.plot(t, us_signal, label="US", color="tab:red", linestyle="--")
    ax_signal.set_title("Stream (last 200 steps)")
    ax_signal.set_ylabel("Indicator")
    ax_signal.legend(loc="upper right")
    ax_signal.grid(True, alpha=0.3)

    palette = ["tab:purple", "tab:green", "tab:orange", "tab:brown", "tab:cyan"]
    for i, demon in enumerate(demons):
        color = palette[i % len(palette)]
        ax_pred.plot(
            t,
            final_predictions[window_start:, i],
            label=f"gamma={demon.gamma}",
            color=color,
        )
    # Overlay the US for visual reference.
    ax_pred.plot(t, us_signal, color="tab:red", linestyle="--", alpha=0.4, label="US")
    ax_pred.set_title("Per-horizon US predictions (last 200 steps)")
    ax_pred.set_ylabel("Prediction")
    ax_pred.legend(loc="upper right")
    ax_pred.grid(True, alpha=0.3)

    # Per-demon squared-error learning curve (smoothed).
    window = 100
    se = result.per_demon_metrics[..., 0]  # (num_steps, n_demons)
    full_t = jnp.arange(num_steps)
    for i, demon in enumerate(demons):
        smoothed = _running_mean(se[:, i], window)
        ax_loss.plot(full_t, smoothed, label=f"gamma={demon.gamma}",
                     color=palette[i % len(palette)])
    ax_loss.set_title(f"Smoothed squared error (window={window})")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_yscale("log")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    plot_path = output_dir / "pavlovian_horde.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pavlovian Horde demo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/step3_pavlovian"),
    )
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cs-us-delay", type=int, default=5)
    parser.add_argument("--iti-min", type=int, default=5)
    parser.add_argument("--iti-max", type=int, default=20)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--gammas",
        type=str,
        default="0.0,0.5,0.9,0.99",
        help="Comma-separated list of gammas",
    )
    args = parser.parse_args()
    gammas = tuple(float(g) for g in args.gammas.split(","))
    main(
        args.output_dir,
        num_steps=args.num_steps,
        seed=args.seed,
        gammas=gammas,
        cs_us_delay=args.cs_us_delay,
        iti_min=args.iti_min,
        iti_max=args.iti_max,
        do_plot=not args.no_plot,
    )

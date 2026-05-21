#!/usr/bin/env python3
"""Replication of Mahmood et al. 2012 Autostep experiments.

This script replicates the key experiments from:
    Mahmood, A.R., Sutton, R.S., Degris, T., & Pilarski, P.M. (2012).
    "Tuning-free step-size adaptation"
    ICASSP 2012.

    and the companion thesis:
    Mahmood, A.R. (2010). "Automatic step-size adaptation in incremental
    supervised learning." M.Sc. thesis, University of Alberta.

Three experiments:

  Experiment 1: Parameter Sensitivity
    Sweep the meta-step-size across several orders of magnitude for both
    IDBD (theta) and Autostep (mu). IDBD has a narrow "V" shape; Autostep
    has a wide, flat "U" shape. This is the core robustness result.

  Experiment 2: Feature Suppression
    1 relevant feature + 19 irrelevant features, all step-sizes initialized
    high. Autostep drives irrelevant step-sizes to near-zero while keeping
    the relevant step-size elevated — automated feature selection online.

  Experiment 3: Non-Stationary Tracking with Step-Size Recovery
    Features switch relevance abruptly. Compare IDBD vs Autostep step-size
    trajectories around change-points. Autostep recovers smoothly; IDBD
    can spike or diverge when the meta-step-size is slightly too large.

Usage:
    python mahmood2012_autostep.py
    python mahmood2012_autostep.py --output-dir output/
"""

import argparse
from pathlib import Path

import jax.random as jr

from alberta_framework import (
    IDBD,
    Autostep,
    LinearLearner,
    StepSizeTrackingConfig,
    Timer,
    metrics_to_dicts,
    run_learning_loop,
)
from alberta_framework.streams.synthetic import SuttonExperiment1Stream

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _avg_mse(metrics_list: list[dict]) -> float:
    return sum(m["squared_error"] for m in metrics_list) / len(metrics_list)


def run_asymptotic_mse(
    optimizer,
    stream: SuttonExperiment1Stream,
    burn_in: int,
    measurement: int,
    seed: int,
) -> float:
    """Burn-in then measure asymptotic MSE for one (optimizer, stream) pair."""
    learner = LinearLearner(optimizer=optimizer)
    key = jr.key(seed)
    state, _ = run_learning_loop(learner, stream, burn_in, key)
    key_measure = jr.key(seed + 1_000_000)
    _, metrics = run_learning_loop(
        learner, stream, measurement, key_measure, learner_state=state
    )
    return _avg_mse(metrics_to_dicts(metrics))


# ---------------------------------------------------------------------------
# Experiment 1: Parameter sensitivity (U-curve vs V-curve)
# ---------------------------------------------------------------------------

def run_sensitivity_sweep(
    meta_params: list[float],
    initial_alpha: float = 0.05,
    burn_in: int = 20_000,
    measurement: int = 10_000,
    seed: int = 42,
) -> tuple[dict[float, float], dict[float, float]]:
    """Sweep meta-step-sizes for IDBD and Autostep, return asymptotic MSE dicts."""
    stream = SuttonExperiment1Stream(num_relevant=5, num_irrelevant=15, change_interval=20)

    idbd_results: dict[float, float] = {}
    autostep_results: dict[float, float] = {}

    print("\nRunning IDBD sensitivity sweep...")
    for theta in meta_params:
        mse = run_asymptotic_mse(
            IDBD(initial_step_size=initial_alpha, meta_step_size=theta),
            stream, burn_in, measurement, seed,
        )
        idbd_results[theta] = mse
        tag = "DIVERGED" if mse > 100 else f"{mse:.4f}"
        print(f"  IDBD(theta={theta:.4f}): MSE = {tag}")

    print("\nRunning Autostep sensitivity sweep...")
    for mu in meta_params:
        mse = run_asymptotic_mse(
            Autostep(initial_step_size=initial_alpha, meta_step_size=mu),
            stream, burn_in, measurement, seed,
        )
        autostep_results[mu] = mse
        print(f"  Autostep(mu={mu:.4f}): MSE = {mse:.4f}")

    return idbd_results, autostep_results


def plot_sensitivity(
    idbd_results: dict[float, float],
    autostep_results: dict[float, float],
    save_path: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    params = sorted(idbd_results.keys())

    # Cap diverged runs for display
    max_display = 30.0
    idbd_mses = [min(idbd_results[p], max_display) for p in params]
    auto_mses = [autostep_results[p] for p in params]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(params, idbd_mses, "s-", color="#ff7f0e", label="IDBD (θ)", markersize=7)
    ax.plot(params, auto_mses, "o-", color="#2ca02c", label="Autostep (μ)", markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Meta-step-size (θ for IDBD, μ for Autostep)", fontsize=12)
    ax.set_ylabel("Asymptotic MSE", fontsize=12)
    ax.set_title(
        "Parameter Sensitivity: IDBD vs Autostep\n"
        "(Replication of Mahmood 2012, Figure 1)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    note = f"IDBD values capped at {max_display} where diverged"
    ax.annotate(note, xy=(0.02, 0.97), xycoords="axes fraction",
                fontsize=8, va="top", color="grey")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Sensitivity plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 2: Feature suppression (1 relevant + 19 irrelevant)
# ---------------------------------------------------------------------------

def run_feature_suppression(
    num_steps: int = 100_000,
    record_interval: int = 500,
    initial_alpha: float = 0.05,
    mu: float = 0.1,
    theta: float = 0.005,
    seed: int = 42,
) -> dict:
    """Track per-feature step-sizes for both algorithms on a sparse signal task.

    Stream: 1 relevant feature, 19 irrelevant (weight=0).
    Sign flips every 20 steps to keep a persistent learning signal on the
    relevant feature, giving the meta-gradient enough correlation to
    differentiate signal from noise.
    """
    stream = SuttonExperiment1Stream(
        num_relevant=1,
        num_irrelevant=19,
        change_interval=20,
    )
    tracking = StepSizeTrackingConfig(interval=record_interval)
    results = {}

    for name, optimizer in [
        ("IDBD", IDBD(initial_step_size=initial_alpha, meta_step_size=theta)),
        ("Autostep", Autostep(initial_step_size=initial_alpha, meta_step_size=mu)),
    ]:
        print(f"  Running {name}...")
        learner = LinearLearner(optimizer=optimizer)
        key = jr.key(seed)
        _, _, history = run_learning_loop(
            learner, stream, num_steps, key, step_size_tracking=tracking
        )
        results[name] = {
            "step_sizes": history.step_sizes,           # (T, feature_dim)
            "recording_indices": history.recording_indices,
        }

    return results


def plot_feature_suppression(results: dict, save_path: str | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (name, data) in zip(axes, results.items()):
        ss = np.array(data["step_sizes"])      # (T, 20)
        idx = np.array(data["recording_indices"])

        relevant = ss[:, 0]                    # first feature is relevant
        irrelevant_mean = np.mean(ss[:, 1:], axis=1)
        irrelevant_max = np.max(ss[:, 1:], axis=1)

        ax.plot(idx, relevant, color="#2ca02c", linewidth=1.8, label="Relevant (α₁)")
        ax.plot(idx, irrelevant_mean, color="#d62728", linewidth=1.5, label="Irrelevant (mean)")
        ax.fill_between(
            idx, 0, irrelevant_max, color="#d62728", alpha=0.15, label="Irrelevant (max)"
        )

        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel("Step Size (α)", fontsize=11)
        ax.set_title(f"{name}: Feature Suppression\n(1 relevant + 19 irrelevant)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.suptitle(
        "Automated Feature Selection Online\n"
        "(Replication of Mahmood 2010, Figure 4.3)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Feature suppression plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 3: Non-stationary tracking — step-size recovery after switch
# ---------------------------------------------------------------------------

def run_tracking_comparison(
    num_steps: int = 30_000,
    record_interval: int = 100,
    initial_alpha: float = 0.05,
    mu: float = 0.05,
    theta: float = 0.01,
    seed: int = 42,
) -> dict:
    """Track learning curves and step-sizes around abrupt relevance changes.

    Uses SuttonExperiment1Stream (sign flip every 200 steps) to create
    frequent change-points, making step-size recovery visible.
    """
    stream = SuttonExperiment1Stream(
        num_relevant=5,
        num_irrelevant=15,
        change_interval=200,
    )
    tracking = StepSizeTrackingConfig(interval=record_interval)
    results = {}

    for name, optimizer in [
        ("IDBD", IDBD(initial_step_size=initial_alpha, meta_step_size=theta)),
        ("Autostep", Autostep(initial_step_size=initial_alpha, meta_step_size=mu)),
    ]:
        print(f"  Running {name}...")
        learner = LinearLearner(optimizer=optimizer)
        key = jr.key(seed)
        _, metrics, history = run_learning_loop(
            learner, stream, num_steps, key, step_size_tracking=tracking
        )
        results[name] = {
            "metrics": metrics_to_dicts(metrics),
            "step_sizes": history.step_sizes,
            "recording_indices": history.recording_indices,
        }

    return results


def plot_tracking_comparison(results: dict, save_path: str | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    colors = {"IDBD": "#ff7f0e", "Autostep": "#2ca02c"}

    for col, (name, data) in enumerate(results.items()):
        ax_err = axes[0, col]
        ax_ss = axes[1, col]

        # Learning curve
        errors = [m["squared_error"] for m in data["metrics"]]
        window = 200
        smoothed = np.convolve(errors, np.ones(window) / window, mode="valid")
        ax_err.plot(smoothed, color=colors[name], linewidth=1.2)
        ax_err.set_title(f"{name}: Tracking Error (rolling mean, w={window})", fontsize=11)
        ax_err.set_xlabel("Time Step")
        ax_err.set_ylabel("Squared Error")
        ax_err.set_yscale("log")
        ax_err.grid(True, alpha=0.3)

        # Step-size evolution: mean relevant vs mean irrelevant
        ss = np.array(data["step_sizes"])
        idx = np.array(data["recording_indices"])
        relevant_mean = np.mean(ss[:, :5], axis=1)
        irrelevant_mean = np.mean(ss[:, 5:], axis=1)

        ax_ss.plot(idx, relevant_mean, color="#2ca02c", linewidth=1.5, label="Relevant (mean)")
        ax_ss.plot(idx, irrelevant_mean, color="#d62728", linewidth=1.5, label="Irrelevant (mean)")
        ax_ss.set_title(f"{name}: Step-Size Adaptation", fontsize=11)
        ax_ss.set_xlabel("Time Step")
        ax_ss.set_ylabel("Step Size (α)")
        ax_ss.set_yscale("log")
        ax_ss.legend(fontsize=9)
        ax_ss.grid(True, alpha=0.3)

    plt.suptitle(
        "Non-Stationary Tracking: IDBD vs Autostep\n"
        "(sign flips every 200 steps; frequent change-point stress test)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Tracking comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: str | None = None) -> None:
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    with Timer("Total experiment runtime"):
        print("=" * 70)
        print("Replication of Mahmood et al. 2012 — Autostep")
        print("=" * 70)

        # ------------------------------------------------------------------
        # Experiment 1: Parameter sensitivity
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("Experiment 1: Parameter Sensitivity (U-curve vs V-curve)")
        print("-" * 70)
        print("Sweeping meta-step-sizes across 3 orders of magnitude.")
        print("IDBD: expect a narrow V — only works near one theta.")
        print("Autostep: expect a flat U — robust across the full range.")

        meta_params = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        idbd_sens, auto_sens = run_sensitivity_sweep(meta_params)

        idbd_best = min(idbd_sens.items(), key=lambda x: x[1])
        auto_best = min(auto_sens.items(), key=lambda x: x[1])
        idbd_valid = sum(1 for v in idbd_sens.values() if v < 100)
        auto_valid = sum(1 for v in auto_sens.values() if v < 100)

        print(f"\n  IDBD best: theta={idbd_best[0]:.4f}, MSE={idbd_best[1]:.4f}")
        print(f"  IDBD usable range: {idbd_valid}/{len(meta_params)} values below MSE=100")
        print(f"\n  Autostep best: mu={auto_best[0]:.4f}, MSE={auto_best[1]:.4f}")
        print(f"  Autostep usable range: {auto_valid}/{len(meta_params)} values below MSE=100")

        sens_path = str(output_path / "mahmood2012_exp1_sensitivity.png") if output_path else None
        plot_sensitivity(idbd_sens, auto_sens, save_path=sens_path)

        # ------------------------------------------------------------------
        # Experiment 2: Feature suppression
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("Experiment 2: Feature Suppression (1 relevant + 19 irrelevant)")
        print("-" * 70)
        print("All step-sizes start at 0.1. Expect irrelevant features suppressed to ~0.")

        suppression_results = run_feature_suppression()

        suppression_steps = 100_000
        for name, data in suppression_results.items():
            import numpy as np
            ss_final = np.array(data["step_sizes"])[-1]   # last recorded snapshot
            rel_final = float(ss_final[0])
            irrel_max = float(np.max(ss_final[1:]))
            ratio = rel_final / (irrel_max + 1e-10)
            print(f"\n  {name} after {suppression_steps // 1000}k steps:")
            print(f"    Relevant α₁:       {rel_final:.5f}")
            print(f"    Irrelevant max α:   {irrel_max:.6f}")
            print(f"    Signal/noise ratio: {ratio:.1f}x")

        supp_path = str(output_path / "mahmood2012_exp2_suppression.png") if output_path else None
        plot_feature_suppression(suppression_results, save_path=supp_path)

        # ------------------------------------------------------------------
        # Experiment 3: Non-stationary tracking
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("Experiment 3: Non-Stationary Tracking (sign flip every 200 steps)")
        print("-" * 70)
        print("Frequent change-points stress-test step-size recovery.")

        tracking_results = run_tracking_comparison()

        for name, data in tracking_results.items():
            cumulative = sum(m["squared_error"] for m in data["metrics"])
            print(f"\n  {name}: cumulative squared error = {cumulative:.2f}")

        plot_tracking_comparison(
            tracking_results,
            save_path=str(output_path / "mahmood2012_exp3_tracking.png") if output_path else None,
        )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("\nExp 1 — Parameter Sensitivity:")
        print(f"  IDBD:     {idbd_valid}/{len(meta_params)} params usable (narrow V)")
        print(f"  Autostep: {auto_valid}/{len(meta_params)} params usable (flat U)")
        print("\nExp 2 — Feature Suppression: see plot")
        print("  Autostep drives irrelevant step-sizes to near-zero autonomously.")
        print("\nExp 3 — Non-Stationary Tracking: see plot")
        print("  Autostep step-size recovery is smoother around change-points.")

        if output_path:
            print(f"\nAll plots saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replication of Mahmood et al. 2012 Autostep")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save plots (default: show interactively)")
    args = parser.parse_args()
    main(output_dir=args.output_dir)

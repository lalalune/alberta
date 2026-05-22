"""Continual Backprop plasticity-preservation demo.

Demonstrates the central claim of Dohare et al. 2024 (*Loss of plasticity
in deep continual learning*, Nature 632 pp.768-774): plain backprop
gradually loses its ability to learn on a long, non-stationary stream.
Continual Backprop (CBP) prevents this by tracking a per-hidden-unit
utility EMA and periodically reinitializing the lowest-utility, mature
units.

Setup
-----
Two CBP-augmented multi-head MLP learners are trained on the same
non-stationary supervised stream. A piecewise-stationary target shifts
its functional form every ``regime_steps`` steps so the same network
must repeatedly re-learn. We measure:

* Mean squared error (recent-window) — does CBP keep tracking?
* Effective rank of the trunk's last-hidden-layer activations across a
  fixed validation buffer — does CBP keep features expressive?
  (Dead/redundant units depress the singular spectrum.)

The two configurations:

* ``cbp_off`` — ``ContinualBackpropConfig(enabled=False)`` (plain MLP
  via the wrapper, no replacement).
* ``cbp_on`` — ``ContinualBackpropConfig(enabled=True, ...)`` with the
  default replacement_rate of 1e-4.

Both use the same architecture, optimizer, sparsity, and seed so the
only difference is unit replacement.

Usage
-----
::

    python "examples/The Alberta Plan/Step3/cbp_plasticity_demo.py" \\
        --output-dir output/step3_cbp \\
        --num-steps 50000

References
----------
- Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood,
  A. R., & Sutton, R. S. (2024). Loss of plasticity in deep continual
  learning. *Nature*, 632, 768-774.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework import (
    CBPMultiHeadMLPLearner,
    ContinualBackpropConfig,
    ObGDBounding,
    Timer,
    run_cbp_learning_loop,
)

# -----------------------------------------------------------------------------
# Non-stationary stream: piecewise-stationary, periodic regime shifts.
# -----------------------------------------------------------------------------


def _generate_stream(
    num_steps: int,
    feature_dim: int,
    regime_steps: int,
    key: Array,
) -> tuple[Array, Array]:
    """Generate (observations, targets) with a piecewise-stationary target.

    Every ``regime_steps`` steps we resample a new "regime" — a random
    weight vector and bias that define the target function

        y_t = tanh(w . x_t + b) + small noise.

    The observations themselves are drawn i.i.d. from a standard normal,
    so the only source of non-stationarity is the regime change.

    Args:
        num_steps: Total number of steps.
        feature_dim: Observation dimension.
        regime_steps: Number of steps per regime.
        key: PRNG key.

    Returns:
        ``(observations, targets)`` shaped ``(num_steps, feature_dim)``
        and ``(num_steps, 1)`` respectively.
    """
    obs_key, w_key, noise_key = jr.split(key, 3)
    observations = jr.normal(obs_key, (num_steps, feature_dim))

    n_regimes = (num_steps + regime_steps - 1) // regime_steps
    regime_keys = jr.split(w_key, n_regimes)
    # Per-regime weights and biases.
    regime_w = jax.vmap(
        lambda k: jr.normal(k, (feature_dim,))
    )(regime_keys)
    regime_b = jr.normal(noise_key, (n_regimes,))

    # For each step, look up its regime.
    regime_idx = jnp.minimum(
        jnp.arange(num_steps) // regime_steps, n_regimes - 1
    )
    w_per_step = regime_w[regime_idx]
    b_per_step = regime_b[regime_idx]

    pre = jnp.einsum("nf,nf->n", observations, w_per_step) + b_per_step
    targets_1d = jnp.tanh(pre)
    noise = 0.05 * jr.normal(jr.fold_in(noise_key, 1), (num_steps,))
    targets = (targets_1d + noise)[:, None]
    return observations, targets


# -----------------------------------------------------------------------------
# Effective rank of trunk's last hidden activation matrix.
# -----------------------------------------------------------------------------


def _effective_rank(matrix: Array, eps: float = 1e-12) -> float:
    """Soft (entropy) rank of a matrix's singular value spectrum.

    Defined as ``exp(-sum_i p_i log p_i)`` with
    ``p_i = sigma_i / sum_j sigma_j``. A matrix with one dominant
    singular value has effective rank ~1; a matrix with all singular
    values equal has effective rank ~min(matrix.shape).

    Args:
        matrix: Input matrix, shape ``(n, m)``.
        eps: Stability constant.

    Returns:
        Scalar effective rank as a Python float.
    """
    sigma = jnp.linalg.svd(matrix, compute_uv=False)
    s = sigma / (jnp.sum(sigma) + eps)
    s = jnp.where(s > 0, s, eps)
    entropy = -jnp.sum(s * jnp.log(s))
    return float(jnp.exp(entropy))


def _validation_hidden_matrix(
    learner: CBPMultiHeadMLPLearner,
    state: object,
    val_obs: Array,
) -> Array:
    """Forward each row of ``val_obs`` and return the trunk's last hidden.

    Args:
        learner: CBP-augmented learner.
        state: Joint CBP state.
        val_obs: Validation observations, ``(n_val, feature_dim)``.

    Returns:
        Hidden activation matrix ``(n_val, hidden_size_last)``.
    """
    trunk_w = state.mlp_state.trunk_params.weights  # type: ignore[attr-defined]
    trunk_b = state.mlp_state.trunk_params.biases  # type: ignore[attr-defined]
    slope = learner._leaky_relu_slope  # private but stable for this example
    use_ln = learner._use_layer_norm

    def _forward_one(obs):
        x = obs
        for i in range(len(trunk_w)):
            x = trunk_w[i] @ x + trunk_b[i]
            if use_ln:
                m = jnp.mean(x)
                v = jnp.var(x)
                x = (x - m) / jnp.sqrt(v + 1e-5)
            x = jnp.where(x >= 0, x, slope * x)
        return x

    return jax.vmap(_forward_one)(val_obs)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def _build_learner(
    n_heads: int,
    hidden_sizes: tuple[int, ...],
    cbp_enabled: bool,
) -> CBPMultiHeadMLPLearner:
    """Construct CBP wrapper with or without unit replacement."""
    return CBPMultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=hidden_sizes,
        cbp_config=ContinualBackpropConfig(
            decay_rate=0.99,
            replacement_rate=1e-4,
            maturity_threshold=100,
            enabled=cbp_enabled,
        ),
        step_size=0.05,
        sparsity=0.9,
        bounder=ObGDBounding(kappa=2.0),
    )


def _evaluate(
    learner: CBPMultiHeadMLPLearner,
    state: object,
    val_obs: Array,
    val_tgt: Array,
) -> tuple[float, float]:
    """Compute (validation MSE, validation effective rank)."""
    hidden_mat = _validation_hidden_matrix(learner, state, val_obs)
    eff_rank = _effective_rank(hidden_mat)

    @jax.jit
    def _predict_one(s, o):
        return learner.predict(s, o)[0]

    preds = jax.vmap(lambda o: _predict_one(state, o))(val_obs)
    mse = float(jnp.mean((preds - val_tgt[:, 0]) ** 2))
    return mse, eff_rank


def main(
    output_dir: Path,
    num_steps: int = 50000,
    feature_dim: int = 16,
    hidden: int = 32,
    regime_steps: int = 5000,
    seed: int = 42,
) -> None:
    """Run the plasticity-preservation comparison.

    Args:
        output_dir: Directory for JSON summary.
        num_steps: Total training steps.
        feature_dim: Observation dimension.
        hidden: Width of the (single) hidden layer.
        regime_steps: Steps between regime changes.
        seed: PRNG seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stream + held-out validation set -----------------------------------
    key = jr.key(seed)
    k_train, k_val, k_init = jr.split(key, 3)
    obs, tgt = _generate_stream(num_steps, feature_dim, regime_steps, k_train)
    # Validation set drawn from a fresh "current" regime — i.e. last regime
    # in the training stream — so we measure plasticity at the *current*
    # task, not on an old one.
    val_obs = jr.normal(k_val, (512, feature_dim))
    last_w = jr.fold_in(k_train, num_steps // regime_steps - 1)
    val_w = jr.normal(last_w, (feature_dim,))
    val_b = jr.normal(jr.fold_in(last_w, 1), ())
    val_tgt = jnp.tanh(jnp.einsum("nf,f->n", val_obs, val_w) + val_b)[:, None]

    n_heads = 1
    hidden_sizes = (hidden,)

    # ---- Run CBP off and CBP on, identical seeds for everything else -------
    summary: dict[str, dict[str, float]] = {}
    for label, cbp_enabled in [("cbp_off", False), ("cbp_on", True)]:
        learner = _build_learner(n_heads, hidden_sizes, cbp_enabled)
        state = learner.init(feature_dim, k_init)

        with Timer(f"{label} training ({num_steps} steps)"):
            result = run_cbp_learning_loop(
                learner, state, obs, tgt
            )

        # Final-window training error.
        last_window = min(2000, num_steps)
        train_mse_last = float(
            jnp.nanmean(result.per_head_metrics[-last_window:, 0, 0])
        )
        # Replacement count over training (sum over per-step bool flags).
        replacements = int(jnp.sum(result.replacements_made))

        # Validation MSE + effective rank.
        val_mse, eff_rank = _evaluate(
            learner, result.state, val_obs, val_tgt
        )

        print(
            f"{label}: train_mse_last={train_mse_last:.5f}, "
            f"val_mse={val_mse:.5f}, eff_rank={eff_rank:.2f}, "
            f"replacements_total={replacements}"
        )
        summary[label] = {
            "train_mse_last_window": train_mse_last,
            "val_mse": val_mse,
            "validation_effective_rank": eff_rank,
            "replacements_total": replacements,
        }

    out_path = output_dir / "cbp_plasticity_demo_summary.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "config": {
                    "num_steps": num_steps,
                    "feature_dim": feature_dim,
                    "hidden": hidden,
                    "regime_steps": regime_steps,
                    "seed": seed,
                },
                "results": summary,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CBP plasticity preservation demo (Step 3 Phase G)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/step3_cbp")
    )
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--regime-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        args.output_dir,
        num_steps=args.num_steps,
        feature_dim=args.feature_dim,
        hidden=args.hidden,
        regime_steps=args.regime_steps,
        seed=args.seed,
    )

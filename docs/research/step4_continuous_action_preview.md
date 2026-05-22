# Step 4 continuous-action actor-critic preview

The Alberta Plan does not single out continuous control, but every continual
agent that interacts with a real-world plant or simulated body needs a way to
emit continuous decisions. Until this preview, the Step 4 control core was
exclusively discrete: ``SARSAAgent`` operates over ``n_actions`` and
``ActorCriticAgent`` parameterises a linear softmax over a finite action set.
This note opens the continuous-action gate with a minimal-but-honest preview
implementation so downstream consumers (rlsecd, security-gym, future Alberta
Plan steps) can begin exercising continuous interfaces.

## API

``alberta_framework.core.actor_critic.ContinuousActorCriticAgent`` mirrors the
existing discrete agent. The configuration replaces ``n_actions`` and
``temperature`` with continuous-action knobs:

```python
ContinuousActorCriticConfig(
    action_dim=1,
    gamma=0.99,
    actor_step_size=3e-5,
    critic_step_size=5e-3,
    actor_lamda=0.9,
    critic_lamda=0.9,
    log_sigma_init=0.0,
    log_sigma_min=-2.0,
    log_sigma_max=1.0,
    action_low=-2.0,
    action_high=2.0,
)
```

The state mirrors the discrete agent's structure but exposes the Gaussian
parameters (mean weights/bias, ``log_sigma``) and per-component eligibility
traces. ``log_sigma`` is per-action-dimension and state-independent in this
preview; the policy is a diagonal Gaussian
``pi(a | s) = N(W_mu s + b_mu, diag(exp(2 log_sigma)))``. Sampled actions are
clipped to ``[action_low, action_high]`` after sampling. The agent supports
``init``, ``policy_params``, ``value``, ``select_action``, ``start``,
``update``, plus ``to_config``/``from_config`` round-tripping. A scan-based
batch loop ``run_continuous_actor_critic_from_arrays`` runs the same update
sequence under ``jax.lax.scan`` for offline replay or batched evaluation.

## Policy gradient

The score function for a diagonal Gaussian ``pi(a | s) = N(mu, diag(sigma^2))``
is well-known and expands per-dimension:

- ``grad_{mu_i} log pi(a | s) = (a_i - mu_i) / sigma_i^2``
- ``grad_{log_sigma_i} log pi(a | s) = (a_i - mu_i)^2 / sigma_i^2 - 1``

These per-dimension gradients enter the actor traces unchanged. The on-policy
AC(lambda) update applies the TD error ``delta_t`` along the traces, which is
identical to the discrete linear AC pattern except that the score function
becomes a Gaussian quantity instead of ``one_hot(A_t) - softmax(logits)``. A
simple diagnostic test verifies the sign: with a positive TD error and
``a > mu``, the mean weights move toward the action.

## Pendulum-v1 5-seed run

The accompanying script
``examples/The Alberta Plan/Step4/continuous_ac_pendulum.py`` runs 5 seeds of
50,000 environment steps on ``Pendulum-v1`` against a uniform-random baseline
on the same seeds. Mean episodic return per 1,000-step window is reported for
both agents.

The latest 5-seed run (saved to
``outputs/continuous_control_preview/pendulum_5seed_results.json``) shows:

| Agent | First window | Final window | Overall mean |
|-------|--------------|--------------|--------------|
| Continuous AC | -1374.9 | -1602.0 | -1566.7 |
| Random uniform | -1220.7 | -1217.7 | -1226.9 |

The continuous AC starts slightly worse than the random baseline, dips through
the early episodes, and then partially recovers (mean over the second half
≈ -1550 vs random's stable -1220). It does not match or beat the random policy
at this horizon and configuration. This is the expected ceiling for a *linear*
diagonal-Gaussian actor: Pendulum-v1 swing-up requires a nonlinear policy
because optimal torque is not a linear function of ``cos(theta), sin(theta),
theta_dot``. A linear policy can only express a fixed gain pattern, which on
this task is dominated by an exploring random walk over the [-2, 2] torque
range.

## Verdict

This preview should remain a **preview**, not be promoted. The
``ContinuousActorCriticAgent`` correctly implements the Gaussian
policy-gradient and trace machinery (eleven unit tests, including a sign
check, trace reset, log-sigma clipping, action clipping, and a config
round-trip), but on Pendulum-v1 the linear actor cannot beat random. To make
continuous control useful for the Alberta Plan we will need:

1. A nonlinear actor (MLP mean head + state-dependent log-sigma head),
   structurally analogous to ``MultiHeadMLPLearner`` for the discrete case.
2. Trunk-trace handling consistent with the rest of the framework
   (``gamma * lamda = 0`` on the shared trunk when hidden layers are present).
3. A Horde-backed continuous critic adapter so the value path can absorb
   auxiliary GVF demons just like ``HordeActorCriticAgent`` does for the
   discrete actor.

The discrete ``ActorCriticAgent`` is unchanged by this preview. If gymnasium
is unavailable the example script writes a skip-summary; the unit tests are
gymnasium-free and therefore always run as part of the pytest suite.

# Step 5/6 Average-Reward Control Primitives

Date: 2026-05-07

This note records the first production average-reward slice for Alberta Plan
Steps 5 and 6.

## What Was Added

`alberta_framework/core/average_reward.py` now provides two temporally
uniform continuing-learning primitives:

- `DifferentialTDLearner`: linear differential TD prediction.
- `DifferentialGTDLearner`: linear off-policy differential GTD/TDC prediction.
- `AverageRewardHordeLearner`: nonlinear shared-trunk differential GVF
  prediction with one learned reward-rate baseline per head.
- `AverageRewardHordeActorCriticAgent`: nonlinear shared-feature
  average-reward actor-critic with a one-head Horde critic and bounded softmax
  actor updates.
- `DifferentialSARSAAgent`: linear epsilon-greedy differential SARSA control.

Both maintain an online reward-rate estimate and update it on every transition.
Both are JAX-compatible, immutable-state implementations with `jax.lax.scan`
array loops:

- `run_differential_td_from_arrays`
- `run_differential_gtd_from_arrays`
- `run_average_reward_horde_from_arrays`
- `run_differential_sarsa_from_arrays`

Production facades were added:

- `alberta_framework/steps/step5.py`
- `alberta_framework/steps/step6.py`

These expose smoke-testable, stable imports through `alberta_framework.steps`
without requiring downstream users to depend directly on exploratory modules.

## Algorithmic Semantics

Step 5 prediction uses differential TD:

`delta_t = R_{t+1} - rbar_t + v(S_{t+1}) - v(S_t)`

`rbar_{t+1} = rbar_t + beta * delta_t`

`e_t = lambda * e_{t-1} + grad v(S_t)`

`theta_{t+1} = theta_t + alpha * delta_t * e_t`

Step 6 control uses differential SARSA:

`delta_t = R_{t+1} - rbar_t + Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)`

The same TD error updates the reward-rate estimate and the action-value
parameters through action-indexed accumulating traces. This is the continuing
average-reward counterpart of Step 4 SARSA.

## Relationship To The Alberta Plan

This closes a primitive gap, not the full Step 5/6 research program.

Closed:

- Continuing prediction without episodic resets or terminal discounts.
- Continuing control with an explicit learned reward-rate baseline.
- Temporally uniform per-step updates.
- JAX scan compatibility and package-level exports.
- Positive-control evidence on closed-form continuing prediction and small
  continuing action-value tasks.
- A seeded two-state continuing-control benchmark where the learned policy must
  condition on state, not only choose a globally better action.
- A seeded off-policy average-reward benchmark where behavior samples the target
  action half the time and importance ratios recover the target policy reward
  rate.
- A seeded nonlinear shared-trunk average-reward Horde benchmark with two GVF
  cumulant heads and per-head reward-rate baselines.
- A seeded nonlinear average-reward actor-critic benchmark where a softmax actor
  learns from the shared-feature Horde critic on a two-state continuing task.

The downstream integration boundary is now covered by a security-gym
integration benchmark that imports the sibling environment API, builds a
synthetic SQLite security stream, and trains Step 6 differential SARSA through
real `SecurityLogStreamEnv` observations and defensive actions.

## Seeded Evidence

The current solution gate consumes six explicit artifacts:

- `outputs/step5_average_reward_prediction/results.json`: 10/10 seeds solve a
  deterministic continuing prediction cycle with true average reward `1.0`;
  mean reward-rate absolute error is `2.5e-6` and centered differential-value
  RMSE is `3.0e-7`.
- `outputs/step5_continuing_control/results.json`: 10/10 seeds solve a
  one-state differential SARSA control task; final-window reward and optimal
  action rate are both `1.0`.
- `outputs/step5_multistate_continuing_control/results.json`: 10/10 seeds solve
  a two-state differential SARSA control task with optimal policy `[1, 0]`;
  final-window reward and policy-match rate are both `1.0`.
- `outputs/step5_off_policy_average_reward/results.json`: 10/10 seeds solve an
  off-policy average-reward prediction task; the behavior reward rate is `0.5`,
  while the learned target-policy reward rate is `0.999997`.
- `outputs/step5_average_reward_horde/results.json`: 10/10 seeds solve a
  nonlinear shared-trunk two-head average-reward Horde task; maximum
  reward-rate absolute error is `2.1e-6`.
- `outputs/step5_average_reward_horde_actor_critic/results.json`: 10/10 seeds
  solve a nonlinear shared-feature average-reward actor-critic task; mean final
  reward is `0.9891` with the correct greedy policy in every seed.

The gate is `benchmarks/step5_solution_gate.py`. With these artifacts present it
reports `solved_step5_full_research_scope=true` for the current local Step 5
average-reward scope. Stage 6 now also has a standalone continuing-control gate:

- `outputs/step6_riverswim/results.json`: 10/10 seeds solve the deterministic
  six-state continuing chain; mean final-window reward is `0.9938` and
  right-action rate is `0.9938`.
- `outputs/step6_riverswim/riverswim_stochastic_results.json`: 10/10 seeds pass
  stochastic RiverSwim; mean final reward is `0.9070` and right-action rate is
  `0.9747`, versus a random baseline of `0.005`.
- `outputs/step6_security_gym/results.json`: 10/10 seeds improve over a
  pass-only security-gym policy; mean evaluation reward improves by `+1.35625`,
  attack alert rate is `0.875`, and benign pass rate is `0.875`.

The Stage 6 gate is `benchmarks/step6_solution_gate.py`. It reports
`accepted_step6_continuing_control=true` for the average-reward
continuing-control completion scope, including deterministic chain, stochastic
RiverSwim, multistate policy control, nonlinear shared-feature actor-critic,
and downstream security-gym integration evidence.

## Acceptance Tests

Focused tests are in:

- `tests/test_average_reward.py`
- `tests/test_step5_step6_production.py`

They verify:

- Exact differential TD-error target semantics.
- Average-reward estimate movement.
- JIT-compatible finite single-step updates.
- JAX scan shapes and finite metrics.
- Differential SARSA config serialization.
- A small continuing two-action positive control where the better action is
  learned.
- Step 5/6 facade smoke tests and public exports.

## Promotion Status

This is the Step 5/6 average-reward and continuing-control completion claim for
the current Alberta Framework scope. It covers local continuing-control
diagnostics plus a downstream security-gym integration check.

# Step 7 Bounded Dyna Planning Primitive

Date: 2026-05-21

This note records the first production planning slice for Alberta Plan Step 7.

## What Was Added

`src/alberta_framework/steps/step7.py` provides a small Dyna-style facade over
existing production primitives:

- Step 6 `DifferentialSARSAAgent` for continuing average-reward control.
- Step 8 `OneStepWorldModel` for online environment prediction.
- `Step7DynaConfig`, `Step7DynaState`, `step7_update`, `run_step7_scan`, and
  `run_step7_smoke`.
- A fixed-size transition memory with bounded, scan-compatible search control.
- Off-policy accounting diagnostics for every imagined backup:
  search-control behavior probability, current target-policy probability, and
  clipped target/behavior ratio.
- Per-decision importance correction for imagined SARSA updates, so Q, trace,
  and reward-rate deltas are scaled by the clipped target/behavior ratio.
- Bounded short-rollout planning through `planning_rollout_depth`; each
  planning budget slot can execute multiple model-generated SARSA backups
  before restoring the real foreground action context.
- Learned search-control utility tracking through `memory_utilities` and
  `planning_utility_step_size`; imagined backups update the selected memory
  item's utility from the observed planning TD signal.
- Six bounded planning strategies:
  - `random`: uniformly sampled action backups.
  - `reward`: choose the action with the largest absolute predicted reward.
  - `surprise`: choose the action with the largest predicted reward plus
    one-step transition magnitude.
  - `predecessor`: choose the stored transition whose successor is closest to
    the current real state, weighted by model prediction priority.
  - `prioritized`: pop the highest-priority stored transition from the bounded
    planning queue, then propagate backup priority to predecessor transitions
    whose successors match the backed-up anchor state.
  - `learned`: select stored transitions by an online utility estimate learned
    from the absolute TD signal produced by previous imagined backups.

Each real transition is handled in this order:

1. Update the world model from the real `(S_t, A_t, R_{t+1}, S_{t+1})`.
2. Update differential SARSA from the same real transition.
3. Store the real transition in the planning memory with its model-prediction
   priority.
4. If the model has passed warmup, spend a fixed `planning_steps` budget on
   model-generated one-step SARSA backups.

The planning backups restore the real `last_observation` and `last_action`
after each imagined update. This keeps the foreground environment interaction
state aligned with the real stream while still allowing model-generated TD
errors to update Q weights and the reward-rate estimate.

## Algorithmic Semantics

Foreground control uses Step 6 differential SARSA:

`delta_t = R_{t+1} - rbar_t + Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)`

Background planning first selects an anchor state from the replay memory, then
selects or samples an action `a`, queries the world model for `(hat R, hat S')`,
and applies the same SARSA update API to the imagined transition. Planning is
accepted only after `planning_warmup_steps` real model updates.
Search-control strategies score stored anchors and all discrete actions under
the learned model, but the total planning budget remains fixed per real step.

## Promotion Status

This closes the local Step 7 planning gap: bounded, temporally uniform,
model-based planning can now run after every real continuing-control update,
with memory-backed search control rather than only last-state planning, an
explicit prioritized-sweeping queue, and off-policy correction for imagined
behavior.

Remaining boundary beyond this local Step 7 completion gate:

- option/subtask discovery variants, which are treated as later planning and
  hierarchy research rather than part of this bounded Dyna landing.

## Acceptance Tests

Focused tests are in `tests/test_step5_step6_production.py`. They verify:

- config serialization roundtrip;
- finite one-step and scan updates;
- warmup-gated planning acceptance;
- random and model-prioritized planning action selection;
- target-policy/search-control probability and importance-ratio accounting;
- importance-ratio scaling of imagined SARSA update deltas;
- prioritized queue pop and recursive predecessor priority propagation;
- short-rollout planning depth greater than one while preserving real context;
- learned search-control anchor selection and utility updates;
- planning output shape stability;
- preservation of the real action context after background planning;
- predecessor search-control anchor selection from stored transitions.

## 2026-05-21 Evidence Gate

Command:

```bash
PYTHONPATH=src python benchmarks/step7_dyna_sample_efficiency.py \
  --seeds 5 \
  --steps 30 \
  --final-window 5 \
  --planning-steps 8 \
  --output outputs/step7_dyna/results.json
```

Result artifact: `outputs/step7_dyna/results.json`.

The benchmark compares Step 6 real-only differential SARSA against Step 7
reward-prioritized one-step Dyna on a continuing one-state, two-action,
average-reward task. Reward is `1` only for action `1`.

Aggregate result:

- real-only final-window reward: `0.920000`
- Step 7 Dyna final-window reward: `1.000000`
- mean reward improvement: `+0.080000`
- mean Q-gap improvement: `+4.794429`
- Q-gap wins: `5/5` seeds
- planning backups accepted: `240` per seed

Gate:

```bash
python benchmarks/step7_solution_gate.py
```

The gate reports
`accepted_step7_dyna_planning_primitive: true` for the bounded Dyna
average-reward control local completion gate.

## Additional Chain Evidence

Command:

```bash
PYTHONPATH=src python benchmarks/step7_numpy_planning.py
```

Result artifact: `outputs/step7_chain_planning/results_numpy.json`.

This tabular diagnostic compares Dyna against real-only control on a six-state
chain over 10 seeds and 500 real steps. Aggregate result:

- real-only cumulative reward: `206.2525`
- Dyna cumulative reward: `292.2500`
- mean cumulative improvement: `+85.9975` (`+41.6953%`)
- cumulative-reward wins: `8/10` seeds
- real-only final-window reward: `0.5999`
- Dyna final-window reward: `0.6609`
- mean convergence speedup to `0.85` reward: `+132.5` steps

## Nonlinear Feature Evidence

Command:

```bash
python benchmarks/step7_nonlinear_feature_planning.py
```

Result artifact: `outputs/step7_nonlinear_feature_planning/results.json`.

This diagnostic uses the same six-state continuing chain, but hides the tabular
state behind dense nonlinear Fourier features and uses a learned feature-space
one-step model for planning. Aggregate result over 10 seeds and 300 real steps:

- real-only final-window reward: `0.78496`
- Dyna final-window reward: `0.88650`
- mean final-window improvement: `+0.10154`
- final-window wins: `6/10` seeds
- mean cumulative improvement: `+26.0810`

## Production Nonlinear JAX Evidence

Command:

```bash
PYTHONPATH=src python benchmarks/step7_production_nonlinear_dyna.py
```

Result artifact: `outputs/step7_production_nonlinear_dyna/results.json`.

This benchmark exercises the promoted `steps.step7` JAX facade with a non-empty
hidden-layer Step 8 world model (`hidden_sizes=(8,)`). Aggregate result over 10
seeds and 30 real steps:

- real-only final-window reward: `0.920000`
- Step 7 Dyna final-window reward: `1.000000`
- mean final-window improvement: `+0.080000`
- mean Q-gap improvement: `+3.232633`
- Q-gap wins: `8/10` seeds
- planning backups accepted: `240` per seed

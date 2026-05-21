# Step 4 Horde Actor-Critic Canonical Search

Date: 2026-05-07

## Decision

Do not promote Horde actor-critic over Q/SARSA yet.

The strongest local conclusion is narrower: the current Horde actor-critic is a
useful Step 3-to-Step 4 integration path, but it is not the canonical control
learner. The promoted Step 4a control default should remain SARSA/Q-style
action-value control until the actor path has a learned nonlinear policy surface
that beats the same baselines on the predefined bsuite gate.

## What Was Tested

This pass compared six control agents on `catch/0` and `cartpole/0`, using
three paired seeds and 2000 continuing steps:

- `autostep_bottleneck`: Q-learning baseline, `(16, 16)` Autostep+ObGD.
- `sarsa_bottleneck`: on-policy SARSA baseline, `(16, 16)` Autostep+ObGD.
- `actor_critic`: existing tuned generic actor-critic path.
- `horde_ac`: existing standalone Horde actor-critic default.
- `horde_ac_tuned`: standalone Horde-AC with lower temperature and separate
  actor-side clipping (`actor_kappa=1e9`, critic `kappa=2.0`).
- `horde_ac_pairwise`: tuned standalone Horde-AC plus causal pairwise feature
  lift for the linear softmax actor.

The pairwise lift is deliberately diagnostic. It applies the same deterministic
map on every time step and uses only the current observation/history vector:
`x -> [x, upper_triangle(x x^T)]`. On `catch/0`, this gives the linear actor
relational features for ball/paddle conjunctions. It is not a learned Step 2
feature-construction mechanism.

## Results

Source artifacts:

- `output/subagents/horde_ac_canonical_search/sweep_3seed_2000/summary.md`
- `output/subagents/horde_ac_canonical_search/sweep_3seed_2000/summary.json`
- `output/subagents/horde_ac_canonical_search/sweep_3seed_2000/paired_improvements.csv`
- `output/subagents/horde_ac_canonical_search/sweep_3seed_2000/horde_ac_control_report.md`

Overall paired improvements are positive when the candidate beats the baseline:

| candidate | vs Q | wins | vs SARSA | wins |
| --- | ---: | ---: | ---: | ---: |
| `actor_critic` | `-30.8333 +/- 39.7592` | 1/6 | `-24.1667 +/- 35.8361` | 2/6 |
| `horde_ac` | `-36.5000 +/- 47.7419` | 2/6 | `-29.8333 +/- 61.7663` | 1/6 |
| `horde_ac_tuned` | `-44.0000 +/- 38.6374` | 1/6 | `-37.3333 +/- 36.8413` | 1/6 |
| `horde_ac_pairwise` | `-42.0000 +/- 33.3661` | 1/6 | `-35.3333 +/- 27.6893` | 1/6 |

By task, `catch/0` is the decisive blocker. Pairwise features reduce default
Horde-AC catch regret slightly (`316.0` mean regret vs `319.3` for default
Horde-AC), but remain far behind Q (`256.0`) and SARSA (`264.0`). On
`cartpole/0`, the pairwise lift hurts relative to the default/tuned branches,
so it cannot be a canonical default.

## Interpretation

The loss is not primarily the old class of discount or episode-boundary bugs:
the bsuite adapters pass discount zero at continuing pseudo-boundaries, the
Horde value head receives explicit transition discounts, and the actor trace is
reset when the discount is zero.

The loss is also not fixed by relaxing actor ObGD. The tuned standalone branch
keeps critic `kappa=2.0` but sets actor `kappa=1e9` and lowers temperature to
`0.5`; this made the 3-seed gate worse overall.

The remaining failure is structural. Q/SARSA receive direct action-value targets
through nonlinear MLP heads. The Horde-AC actor receives only sampled-action
policy-gradient feedback from a scalar Horde value head, and the core policy is
linear in the adapter features. Pairwise features partially help catch, which
supports the actor-capacity diagnosis, but the fixed lift is slower and less
robust than learned action-value heads.

## Code Changes

- `benchmarks/bsuite/agents/horde_actor_critic.py`
  - Added `actor_kappa` so actor clipping can be ablated separately from critic
    clipping.
  - Added `feature_lift={"raw", "quadratic", "pairwise"}` with a
    `max_feature_dim` guard.
- `benchmarks/bsuite/configs.py`
  - Added `horde_ac_tuned` and `horde_ac_pairwise`.
- `scripts/step4_horde_ac_sweep.py`
  - Added a reproducible paired sweep/report script.
- `tests/test_horde_actor_critic.py`
  - Added focused regression tests for the pairwise feature lift and adapter
    initialization.

## Reproduction

```bash
source .venv/bin/activate
python scripts/step4_horde_ac_sweep.py \
  --save-path output/subagents/horde_ac_canonical_search/sweep_3seed_2000 \
  --num-steps 2000 \
  --seeds 0 1 2
```

Validation:

```bash
source .venv/bin/activate
pytest tests/test_horde_actor_critic.py tests/test_bsuite_agents.py -q
```

## Next Blocker

The next promotable candidate should not be another fixed feature expansion.
It should implement a nonlinear learned actor that shares or mirrors the Step 2
feature path:

1. Shared-trunk Horde actor-critic where the actor reads the same learned hidden
   features as the value/GVF critic, with actor gradients routed through a
   policy head.
2. A fair MLP actor baseline with matched optimizer, bounder, normalization,
   hidden sizes, and per-step temporal-uniform updates.
3. Advantage normalization or a learned average-reward/value baseline only after
   the actor has nonlinear features; the current sweep suggests clipping and
   temperature are not sufficient.
4. Promotion gate: beat both `autostep_bottleneck` and `sarsa_bottleneck` on
   paired `catch/0` and `cartpole/0` 10-seed, 2000-step runs before expanding to
   broader bsuite families.

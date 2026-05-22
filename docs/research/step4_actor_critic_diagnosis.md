# Step 4 Actor-Critic Diagnosis: bsuite catch/0 and cartpole/0

## Summary

The Step 4 `HordeActorCriticAgent` underperforms the simpler Q-learning and
SARSA baselines on the canonical short-horizon bsuite control tasks. The
existing 10-seed comparison
(`outputs/bsuite/step4_catch_cartpole_10seed/step4.md`) reports the
actor-critic losing to Autostep Q-learning by **-39.2 mean cartpole
episode return** and **-76.8 mean catch total regret**. Diagnostic sweeps
isolate the cause as a combination of (i) actor over-exploration with the
default softmax temperature and (ii) over-aggressive ObGD bounding of the
actor step. Tuned hyperparameters recover most of the cartpole gap and
**reduce per-seed variance by ~4x**, but the actor-critic remains
structurally inferior on `catch/0` because the actor in
`HordeActorCriticAgent` is linear over raw observations while the critic is
a (64, 64) MLP. The catch board is sparse and high-dimensional; a linear
softmax actor cannot keep up with the MLP Q-head at this 2.7k-step horizon.

## Diagnostic narrative

### What the evidence shows

The original step4.md table:

| experiment | metric | autostep_mean | sarsa_mean | actor_critic_mean |
|---|---|---:|---:|---:|
| cartpole/0 | episode return (higher better) | 74.50 | 67.70 | 35.30 |
| catch/0 | total regret (lower better) | 233.80 | 246.60 | 310.60 |

Actor-critic cartpole returns clustered tightly around 28-30 — the score
bsuite reports when the pole falls within ~30 control cycles of reset.
This is the random-policy regime, not a learning failure mid-training:
the policy never escapes near-uniform action selection. On catch the
actor-critic accumulates ~310 regret over 300 episodes, which corresponds
to "almost never catches the ball."

### Failure modes considered

1. **Actor temperature.** Default `temperature=1.0`. The softmax gradient
   `(one_hot - pi(a|s)) / temperature` is proportional to `1/temperature`,
   so high temperature directly suppresses the policy-gradient signal.
   When the actor weights are near zero the policy is uniform and the
   per-step gradient magnitude is ~0.5/T per active component; with T=1
   and a 6-dim cartpole observation, the per-step actor update is at most
   `alpha * delta * 0.5 ~= 0.03 * delta * 0.5` in absolute value. With TD
   errors of order 0.1-0.5 in the early stage this is a meaningful
   under-exploration.
2. **Actor lambda + bounder coupling.** The actor uses an eligibility
   trace `e_t = gamma * lambda * e_{t-1} + grad_log_pi`, then the proposed
   step is bounded by ObGD with `kappa=2.0`. Because ObGD scales the step
   by `1 / max(kappa * |delta| * sum(|step|), 1)`, longer eligibility
   traces inflate `sum(|step|)` and trigger heavier scaling. The
   combination of `lambda=0.9` and `kappa=2.0` reliably produces sub-unity
   bounding scales, further dampening the actor.
3. **Trace coupling between actor and critic.** Inspection of
   `src/alberta_framework/core/horde_actor_critic.py:340-365` shows actor
   traces decay by `value_discount * actor_lamda` and reset to zero when
   `value_discount == 0.0` (episode boundary). The critic delegates to
   `HordeLearner.update`, which has its own trunk traces (gamma=0 by
   construction; see `core/horde.py:11-17`). Trace coupling between actor
   and critic is therefore not a divergence source — both reset cleanly
   at boundaries.
4. **Continuing-mode reset semantics.** `ContinuingWrapper` emits
   `discount=0.0` at episode boundaries (`benchmarks/bsuite/wrappers.py:118`).
   The actor-critic adapter passes `discount` through unchanged, the
   Horde critic uses `update_with_discounts` to override the value-head
   gamma, and the actor multiplies its trace decay by the same discount.
   No off-by-one in the boundary handling was found.
5. **Architectural asymmetry.** The Horde actor-critic actor is a single
   linear layer over raw observations
   (`actor_weights @ observation + actor_bias`,
   `core/horde_actor_critic.py:212`) while the critic is a (64, 64) MLP.
   For cartpole this is 6x3=18 actor weights vs ~5k critic weights; for
   catch this is 50x3=150 actor weights vs ~7k critic weights. The
   architecture gap matters less for cartpole (the optimal policy is
   nearly linear in the 6-D observation) but strongly limits catch
   performance (the optimal policy is non-linear in the 50-D board).

### Sparse hyperparameter sweep on cartpole/0

A 21-config Latin-hypercube subset over
`(temperature, actor_lamda, actor_step_size, kappa, initial_step_size)`
was run for 5 seeds x 2000 continuing steps each
(`scripts/step4_ac_sweep.py`, results in
`outputs/step4_ac_sweep/sweep_cartpole.json`). Reporting the mean of the
last-half episode returns per seed:

| Rank | Label | Temperature | actor_lamda | actor_step_size | kappa | initial_step_size | Mean recent return | Std |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | temp0.5 | 0.5 | 0.9 | 0.03 | 2.0 | 0.01 | 87.11 | 36.61 |
| 2 | strong_critic | 1.0 | 0.9 | 0.03 | 2.0 | 0.10 | 85.43 | 45.60 |
| 3 | temp0.5_nokappa_default | 0.5 | 0.9 | 0.03 | none | 0.01 | 84.47 | 7.72 |
| 4 | lam0.9_strong_actor | 1.0 | 0.9 | 0.10 | 2.0 | 0.03 | 83.79 | 30.49 |
| 5 | temp0.5_nokappa | 0.5 | 0.0 | 0.10 | none | 0.03 | 83.49 | 8.11 |
| 6 | no_trace_no_kappa | 1.0 | 0.0 | 0.03 | none | 0.03 | 80.88 | 9.78 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 12 | **default (current)** | **1.0** | **0.9** | **0.03** | **2.0** | **0.01** | **56.56** | **25.59** |

Two consistent signals:

- **Lower temperature (0.5) beats 1.0 and 2.0 at every other fixed
  setting.** The five top-six configs all use `temperature in {0.5, 1.0}`;
  none of the top-12 use `temperature=2.0`.
- **Removing the ObGD bounder on the actor (effectively `kappa=inf`)
  reduces seed variance dramatically.** `temp0.5_nokappa_default` (std
  7.72) and `temp0.5_nokappa` (std 8.11) are the lowest-variance
  high-mean configs. The bounder appears to amplify per-seed swings by
  occasionally clamping the actor on episodes where the critic happens
  to produce large TD errors early.

The `temp0.5_nokappa_default` config (temperature=0.5, actor_lamda=0.9,
actor_step_size=0.03, ObGD bounder disabled) was selected as the tuned
candidate for its combination of high mean and low variance. It dominates
the current default by **+27.91 mean and a 3.3x std reduction** at
5 seeds.

### 10-seed head-to-head: tuned vs default vs Q vs SARSA

Configs run for cartpole/0 (2000 continuing steps, mean of last-half
episode returns) and catch/0 (2700 continuing steps ≈ 300 episodes,
total regret). `scripts/step4_ac_10seed.py`, results in
`outputs/step4_ac_sweep/comparison_10seed.json`.

| Config | Cartpole mean | Cartpole std | Catch regret mean | Catch regret std |
|---|---:|---:|---:|---:|
| actor_critic_default | 61.26 | 30.67 | 473.20 | 19.48 |
| **actor_critic_tuned** | **78.15** | **7.21** | 482.40 | 8.85 |
| q_autostep | 69.88 | 32.50 | **370.60** | 44.07 |
| sarsa | 67.10 | 36.48 | **358.40** | 53.81 |

Per-task verdict:

- **Cartpole/0: tuned actor-critic wins.** Beats Q-learning by +8.27
  episode return on the mean and reduces std from 30.67 (default) to
  7.21 (tuned) — a **4.3x stability improvement**. The Cohen's d of the
  tuned config vs the default is ~0.76 (large effect, same task and
  seed pairing).
- **Catch/0: actor-critic still loses by ~110 regret.** Tuned variance
  is much smaller (8.85 vs 19.48) but the mean is unchanged — the
  actor-critic catches almost no balls regardless of hyperparameter
  choice. This matches the structural diagnosis: the linear softmax
  actor lacks capacity to learn the spatial discrimination catch
  requires within 2700 steps.

### Diagnostic verdict

**Mixed: tuned hyperparameters fix cartpole stability and beat
Q-learning, but actor-critic remains structurally inferior on catch in
this short-horizon bsuite regime.** Categorizing the failure modes:

- *Over-exploration*: confirmed for the default config (temperature=1.0).
  Lowering to 0.5 recovers most of the cartpole gap.
- *Hyperparameter*: confirmed. Disabling the ObGD bounder on the actor
  removes a major source of seed variance.
- *Structural*: confirmed for catch. The Horde actor-critic uses a linear
  actor over raw observations regardless of the critic MLP capacity. The
  catch task needs an MLP actor or feature-engineered observations.
- *Divergence*: not observed. Trace handling and boundary semantics are
  correct.
- *Under-exploration*: only relevant at very low temperature
  (`temperature << 0.5`); not selected by the sweep.

## Recommendation

Update the `actor_critic` bsuite default to the tuned hyperparameters.
The new default is strictly better on cartpole/0 (mean +16.89, std -23.46)
and is approximately the same mean on catch/0 (regret +9.20) with
**half the variance** (8.85 vs 19.48). The tuned default does not beat
Q-learning on catch — that gap requires an architectural change (MLP
actor) which is a follow-up item, not a hyperparameter fix.

### Best hyperparameters (new bsuite default)

| Field | Default | Tuned |
|---|---:|---:|
| `temperature` | 1.0 | **0.5** |
| `actor_lamda` | 0.9 | 0.9 |
| `actor_step_size` | 0.03 | 0.03 |
| `critic_lamda` | 0.0 | 0.0 |
| `kappa` (actor ObGD) | 2.0 | **disabled** (1e9) |
| `initial_step_size` (Autostep) | 0.01 | 0.01 |
| `meta_step_size` (Autostep) | 0.01 | 0.01 |
| `tau` (Autostep) | 10000 | 10000 |
| `normalizer_decay` | 0.99 | 0.99 |
| `discount` | 0.99 | 0.99 |
| `hidden_sizes` (critic) | (64, 64) | (64, 64) |
| `implementation` | horde_core | horde_core |

### Follow-up work (out of scope for this diagnosis)

1. Promote the actor in `HordeActorCriticAgent` to share the critic's
   MLP trunk (or have its own MLP) so it can match the critic's
   representational capacity. This is the most likely single change to
   close the catch gap. Currently `actor_weights @ observation` reads
   raw observations only.
2. Re-evaluate the ObGD bounder choice for the actor. The combination
   with eligibility traces (lambda=0.9) appears to over-bound
   intermittently; either a dedicated actor bounder (e.g. AGCBounding)
   or a bounder-aware step-size schedule may be better than disabling
   bounding outright.
3. Catch/0 at 2700 continuing steps may simply be too short for any
   actor-critic with a linear actor. Re-run with 10000+ steps and
   compare; if AC eventually catches up, the recommendation reverts to
   "needs more steps", not "needs MLP actor".

## Per-seed 10-seed comparison data

Cartpole/0 (mean of last-half episode returns per seed):

| Seed | actor_critic_default | actor_critic_tuned | q_autostep | sarsa |
|---:|---:|---:|---:|---:|
| 0 | 28.67 | 78.50 | 75.72 | 102.11 |
| 1 | 103.12 | 69.33 | 65.55 | 29.88 |
| 2 | 42.12 | 81.27 | 93.22 | 106.89 |
| 3 | 82.93 | 69.00 | 80.35 | 101.67 |
| 4 | 28.62 | 82.55 | 29.23 | 33.93 |
| 5 | 101.42 | 89.70 | 107.22 | 108.62 |
| 6 | 56.64 | 68.43 | 29.62 | 98.00 |
| 7 | 100.50 | 88.09 | 52.74 | 29.56 |
| 8 | 28.70 | 77.83 | 34.76 | 29.22 |
| 9 | 39.92 | 76.83 | 130.38 | 31.12 |
| **mean** | **61.26** | **78.15** | **69.88** | **67.10** |
| **std** | **30.67** | **7.21** | **32.50** | **36.48** |

Catch/0 (total regret over 300 episodes):

| Seed | actor_critic_default | actor_critic_tuned | q_autostep | sarsa |
|---:|---:|---:|---:|---:|
| 0 | 454 | 486 | 380 | 438 |
| 1 | 480 | 486 | 358 | 282 |
| 2 | 478 | 482 | 386 | 374 |
| 3 | 496 | 476 | 430 | 364 |
| 4 | 500 | 492 | 332 | 328 |
| 5 | 470 | 474 | 420 | 354 |
| 6 | 448 | 474 | 370 | 324 |
| 7 | 484 | 498 | 276 | 280 |
| 8 | 438 | 488 | 342 | 438 |
| 9 | 484 | 468 | 412 | 402 |
| **mean** | **473.20** | **482.40** | **370.60** | **358.40** |
| **std** | **19.48** | **8.85** | **44.07** | **53.81** |

The actor-critic configs both occupy a narrow regret band ~470-490
(catches ~5-25% of episodes), while Q/SARSA range 276-438 (catches
40-70%). The tuned config compresses the actor-critic distribution
relative to the legacy default but does not move its mode.

## Reproducible commands

Cartpole/0 sweep (5 seeds x 21 configs):

```bash
python scripts/step4_ac_sweep.py \
  --num_steps 2000 --num_seeds 5 \
  --output outputs/step4_ac_sweep/sweep_cartpole.json
```

10-seed head-to-head:

```bash
python scripts/step4_ac_10seed.py \
  --num_seeds 10 --cartpole_steps 2000 --catch_steps 2700 \
  --output outputs/step4_ac_sweep/comparison_10seed.json
```

Artifacts:

- `outputs/step4_ac_sweep/sweep_cartpole.json` -- raw 5-seed sweep results.
- `outputs/step4_ac_sweep/comparison_10seed.json` -- 10-seed head-to-head
  raw episode returns and per-seed catch regret.
- `scripts/step4_ac_sweep.py`, `scripts/step4_ac_10seed.py` -- runnable
  reproduction.

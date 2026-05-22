# Self-Simulation, Dreaming, and Dyna Integration

Status: scoped interface plus latent selection probe, not a promoted planning
learner.

## Summary

The intuition is right: if the agent only updates value, policy, and features
from the action it just took, then it is using the environment as a label
source but not yet learning a usable model of the environment. A world model
adds a second supervised problem: predict `observation_{t+1}`, reward,
continuation/discount, and optionally model confidence from
`(observation_t, action_t)`. A behavior model adds the complementary problem:
predict or sample the agent's own future actions from imagined observations.
Together they can generate imagined transitions for bounded Dyna-style
planning.

The implemented lane is deliberately small. It provides
`src/alberta_framework/core/dreaming.py`, a protocol-based rollout interface
plus a guarded adapter for the current action-conditioned world model. It can
consume the current `core/world_model.py`, `core/behavior_model.py`, and
`core/working_memory.py` surfaces without owning their internals.

As of the current pass, the concrete environment-prediction side is
`ActionConditionedWorldModel`, which predicts next-observation delta, reward,
and discount from `(observation, action)`. The first runnable evidence harness is
`examples/The Alberta Plan/Step3/action_conditioned_cartpole_dynamics.py`; it
tests whether linear and MLP predictors can learn random-policy CartPole
dynamics before any Dyna control claim is made. The initial 10-seed, 20k-step
CartPole sweep selects the linear delta model as the current one-step dynamics
candidate: final-window next-observation RMSE `0.00845` versus `0.01363` for the
MLP condition.

The LeWorldModel follow-up adds a low-dimensional latent version of the same
idea in `src/alberta_framework/core/latent_world_model.py`. It is inspired by
LeWM's action-conditioned JEPA framing, but it is not yet the full pixel
architecture: the encoder is fixed, the predictor learns latent delta, reward,
and discount online, and the anti-collapse terms are diagnostics/gates rather
than representation-learning losses. The useful new piece is selective dream
search: `DreamSelectionConfig` ranks real or imagined candidates by surprise,
utility, behavior-model confidence, and model-error penalties so planning does
not train on every model-generated transition.

## Where It Attaches

The current production path in `src/alberta_framework/pipeline.py` is already
transition-oriented:

1. Step 2 converts a raw observation into features.
2. Step 3 updates Horde predictions from `(features_t, cumulants_t,
   features_{t+1})`.
3. Step 4 updates control from the same transition.
4. Step 5/6 average-reward primitives update from continuing transitions.

Dreaming should attach after each real transition update:

1. Foreground: observe `o_{t+1}`, update Step 2, Step 3, Step 4, and the world
   and behavior models from the real transition.
2. Background: spend a fixed or bounded compute budget generating imagined
   transitions from a current feature/observation seed.
3. Background: convert imagined transitions into ordinary learner items and
   call the same update APIs used by real transitions, with model-confidence
   weights or gates.

This preserves temporal uniformity because there is no special training phase.
Every real step may receive the same bounded planning budget. Reactive action
selection still happens before background planning, matching the Alberta Plan's
foreground/background distinction.

## Implemented Surface

`dreaming.py` defines:

- `GuardedDreamer` and `DreamingConfig`: a one-step guard for
  `ActionConditionedWorldModel` proposals using warmup, model-error EMA,
  uncertainty, finiteness, and discount clamps.
- `RecentObservationBuffer`: a fixed-size ring buffer for anchoring dreams in
  recently observed states rather than arbitrary out-of-support states.
- `DreamWorldModel` protocol: `predict(state, observation, action, key)`.
- `DreamBehaviorModel` protocol: `sample_action(state, observation, key)`.
- `DreamRolloutConfig`: rollout horizon, confidence threshold, model-error
  limit, discount floor, and terminal handling.
- `ActionConditionedDreamWorld`: adapter from the current
  `ActionConditionedWorldModel` into the rollout protocol.
- `DreamRolloutState`: current imagined observation, PRNG key, active flag,
  cumulative confidence, and step count.
- `dream_one_step`: generate one imagined transition without mutating model or
  real-environment state.
- `dream_rollout`: scan-compatible short rollout.
- `imagined_transition_to_supervised_item`: build model-learning targets from
  imagined transitions.
- `imagined_transition_to_gvf_item` and `imagined_rollout_to_gvf_items`: build
  Horde/GVF-style transition arrays.
- `imagined_rollout_to_sarsa_items`: build SARSA-style arrays with shifted next
  actions; the final transition is masked unless a bootstrap action is supplied.
- `DreamSelectionConfig` and `score_dream_candidates`: bounded search control
  for selecting surprising/useful/confident imagined updates.
- `BehaviorModelDreamPolicy`: adapter from the supervised behavior model to
  the dream-rollout action protocol.

The API is JAX-compatible in the same sense as the rest of the framework:
protocol implementers must use pure functions over immutable model state, and
rollouts use `jax.lax.scan`.

## Existing Research

- Sutton's Dyna architecture integrates learning, planning, and reactive
  execution, with planning performed from learned models while action selection
  remains reactive. See Sutton 1991,
  [Dyna, an Integrated Architecture for Learning, Planning and Reacting](https://papersdb.cs.ualberta.ca/~papersdb/view_publication.php?pub_id=500).
- Moore and Atkeson's
  [prioritized sweeping](https://www.ri.cmu.edu/publications/prioritized-sweeping-reinforcement-learning-with-less-data-and-less-real-time/)
  adds a search-control mechanism that uses previous experience to prioritize
  dynamic-programming backups.
- Lin's 1992
  [self-improving reactive agents](https://www.cs.utexas.edu/~shivaram/readings/b2hd-Lin1992.html)
  studied experience replay, action models, and teaching in neural RL agents.
- Imagination-Augmented Agents use predictions from a trained environment model
  as additional context for deep policies:
  [Racaniere et al. 2017](https://papers.nips.cc/paper/7152-imagination-augmented-agents-for-deep-reinforcement-learning).
- Dreamer learns latent world models and learns behavior from imagined latent
  trajectories:
  [Hafner et al. 2020](https://dreamrl.github.io/).
- LeWorldModel trains an action-conditioned JEPA-style world model from pixels
  with a next-embedding prediction loss plus a Gaussian latent regularizer, and
  uses latent surprise to detect physically implausible events:
  [Maes et al. 2026](https://arxiv.org/abs/2603.19312).
- MuZero combines tree search with a learned model that predicts planning-
  relevant rewards, values, and policies rather than reconstructing full
  observations:
  [Schrittwieser et al. 2020](https://www.nature.com/articles/s41586-020-03051-4).
- The Alberta Plan explicitly places planning in the background while
  perception, value learning, and reactive policy updates remain foreground,
  temporally uniform processes:
  [Sutton, Bowling, and Pilarski 2023](https://arxiv.org/abs/2208.11173).

## What Is Known

Dyna and prioritized sweeping already establish the core pattern: learn a model
from real experience, generate simulated transitions, and feed those transitions
to otherwise ordinary prediction/control updates. Modern world-model agents
show that latent imagination can scale when the model is accurate enough and
when planning is tightly coupled to value/policy learning.

The hard known problem is model bias. In online continual settings, a learned
model can drift, extrapolate outside its data support, or overwrite newly useful
features. Planning on those errors can accelerate failure. This is why the first
interface includes `confidence` and `model_error` gates and treats generated
items as weighted training data, not ground truth.

## What Is Novel Here

The local novelty is not "dreaming" itself. The contribution is a small
Alberta-framework contract:

- the same imagined transition object can feed supervised model learning,
  Horde/GVF prediction, and SARSA-style control;
- model use is bounded and can be called after every real step;
- world and behavior models are separate, so the agent can explicitly learn
  both environment response and its own action distribution;
- confidence/error gates are first-class fields so later resource-management
  and utility diagnostics can decide when dreaming pays rent.

## Acceptance Criteria Before Promotion

This interface is ready for Worker E/F integration, but not enough for a Step
7/8 claim. Promotion requires:

1. A concrete world model that learns online from real transitions and reports
   calibrated confidence or model-error estimates.
2. A concrete behavior model that mirrors the current policy or learns the
   agent's action distribution without a special offline phase.
3. A Dyna update loop that calls real updates first, then a bounded number of
   imagined updates using the same learner APIs.
4. Ablations: no dreaming, replay of real transitions, one-step Dyna,
   short-rollout Dyna, confidence-gated Dyna, and prioritized sweeping.
5. Diagnostics: model prediction error, generated-update acceptance rate,
   dream depth survival, effect on real online return/MSE, and failure cases
   where dreaming hurts.
6. Off-policy accounting when imagined behavior diverges from the target
   policy, ideally through the off-policy Horde backend.

The next concrete promotion target is a one-step Dyna SARSA/Horde actor-critic
experiment that uses the CartPole dynamics predictor only after its
final-window prediction error falls below a predeclared gate.

## Current Limitations

The implementation does not learn a model, search a tree, prioritize backups,
or perform Dreamer/MuZero-style latent value-gradient training. It is a narrow
Dyna transition generator. That narrowness is intentional: it gives us a stable
integration point while keeping the scientific question testable.

## 2026-05-07 Probe Results

Command:

```bash
python "examples/The Alberta Plan/Step8/model_memory_dreaming_probe.py" \
  --output-dir output/model_memory_dreaming_probe_counterfactual_d2_30seed \
  --seeds 30 \
  --steps 600 \
  --final-window 100 \
  --dreams-per-step 2 \
  --dream-warmup 80 \
  --dream-action-mode counterfactual
```

This controlled probe separately checks behavior prediction, working memory,
and guarded Dyna updates.

Results:

- Behavior model final NLL: `0.44556 +/- 0.01472`; uniform-policy NLL
  improvement: `+0.24759`.
- Behavior final-window accuracy: `0.79467`.
- Working-memory delayed-action MSE: raw `0.25000`, memory `0.00000`,
  `30/30` wins.
- Behavior-sampled dreams were negative in the same probe family. At
  `dreams_per_step=4`, real-only reward MSE was `0.04819` and behavior-sampled
  Dyna MSE was `0.05063`.
- Counterfactual dreams were positive. At `dreams_per_step=2`, real-only reward
  MSE was `0.04436`, real + dream MSE was `0.04312`, improvement was
  `+0.00123`, and Dyna won `22/30` seeds.

Interpretation: the first useful dreaming signal comes from counterfactual
action coverage, not from replaying behavior-like imagined transitions. This
supports the user's hypothesis in a specific way: predicting the environment is
valuable when it is used to ask "what if I took another action here?", while a
behavior model is still useful for likelihoods, off-policy accounting, and
behavior-like rollouts.

Promotion status: not yet promoted as a control result. The positive Dyna probe
is reward-prediction sample-efficiency evidence on a controlled system. The next
promotion experiment should attach counterfactual, model-error-weighted dreams
to a real Horde/SARSA or actor-critic update and measure online return.

## 2026-05-07 Longer-Horizon Follow-up

A stricter 800-step, 50-seed confirmation did not promote the initial
counterfactual dreaming setting:

```bash
python "examples/The Alberta Plan/Step8/model_memory_dreaming_probe.py" \
  --output-dir output/model_memory_dreaming_probe_counterfactual_d2_50seed_800 \
  --seeds 50 \
  --steps 800 \
  --final-window 150 \
  --dreams-per-step 2 \
  --dream-warmup 100 \
  --dream-action-mode counterfactual
```

Results:

- Behavior model final NLL: `0.45003 +/- 0.00766`; uniform-policy NLL
  improvement: `+0.24312`.
- Working-memory delayed-action MSE remained exact: raw `0.25000`, memory
  `0.00000`, `50/50` wins.
- Guarded counterfactual Dyna was mean-negative: real-only reward MSE
  `0.03565`, real + dream MSE `0.04082`, improvement `-0.00516`, despite
  `27/50` seed wins.

Safety sweeps at 800 steps also stayed negative:

- `dream_step_size=0.003`, `dreams_per_step=2`, 30 seeds:
  improvement `-0.00471`, `19/30` wins.
- `dream_step_size=0.0015`, `dreams_per_step=2`, 30 seeds:
  improvement `-0.00244`, `20/30` wins.
- `dream_step_size=0.003`, `dreams_per_step=1`, 30 seeds:
  improvement `-0.00253`, `17/30` wins.
- Loss-pressure gates with thresholds `0.001`, `0.003`, and `0.006` were also
  negative in 10-seed probes.

Interpretation: the current dream updates can help early sample efficiency, but
continued imagined reward updates inject enough model bias to lose at a longer
horizon. This is useful negative evidence. The canonical promotion should keep
environment prediction, behavior prediction, working memory, confidence
diagnostics, and the Dyna interface, but it should not promote the current
fixed one-step reward-dream update as a default learner mechanism.

## 2026-05-09 Action-Interaction and Oracle Follow-up

The longer-horizon failure was traced to a structural reward-model issue. The
probe reward contains an `x * action` term, while the first linear world model
only received `[x, one_hot(action)]`. The real reward learner had the
interaction feature, but the learned world model did not, so counterfactual
dream rewards were biased.

The world model now supports optional observation-by-action interaction inputs,
and the Step 8 probe enables them by default.

Results at 800 steps:

- Interaction world model, 10 seeds: real-only reward MSE `0.04021`, real +
  dream MSE `0.03800`, improvement `+0.00221`, `7/10` wins.
- Interaction world model, 30 seeds: real-only reward MSE `0.03211`, real +
  dream MSE `0.03514`, improvement `-0.00302`, despite `21/30` wins.
- Reward-error gates at thresholds `0.03`, `0.05`, `0.08`, and `0.12` did not
  fix the 30-seed mean.
- Higher world-model step sizes reduced model error but did not close the
  learned-reward gap. At `world_step_size=0.10`, 30 seeds gave improvement
  `-0.00469`, despite `22/30` wins.
- Oracle imagined rewards were decisive: real-only reward MSE `0.03211`, oracle
  real + dream MSE `0.02229`, improvement `+0.00982`, `30/30` wins.

Interpretation: counterfactual Dyna is valuable when the imagined reward target
is accurate. The remaining blocker is learned reward calibration under
counterfactual action queries, not the planning update itself. The next
canonical improvement should learn calibrated uncertainty or ensembles for the
reward head, or split the world model into separately calibrated next-state,
reward, and discount predictors before allowing imagined updates to affect
downstream learners.

## 2026-05-09 Simplification: Calibrated Reward Model

The smallest positive mechanism is not generic dreaming from a monolithic
world model. It is counterfactual reward supervision from a fast calibrated
scalar reward model.

The decisive ablations at 800 steps were:

- Multi-head world-model reward, 30 seeds: improvement `-0.00302`, `21/30`
  wins.
- Separate LMS reward model, 30 seeds: improvement `+0.00007`, `18/30` wins.
  This is positive but not compute-worthy.
- Separate RLS reward model, one dream per step, 30 seeds: improvement
  `+0.00369`, `30/30` wins.
- Separate RLS reward model, two dreams per step, 30 seeds: improvement
  `+0.00780`, `30/30` wins.
- Oracle imagined rewards, two dreams per step, 30 seeds: improvement
  `+0.00982`, `30/30` wins.

Theory sketch:

Let `g_t(s, a)` be the downstream supervised update direction induced by an
imagined reward target and let `r_hat_t(s, a) = r(s, a) + b_t(s, a)` be the
learned reward model. A real-only learner avoids model bias but only samples
the behavior distribution. Counterfactual Dyna is useful when the reduction in
counterfactual coverage error is larger than the bias term introduced by
`b_t`. In this probe, oracle targets show that the coverage term is strongly
positive. The monolithic world-model reward head and LMS reward model leave
too much counterfactual reward bias. RLS reduces that bias fast enough that the
net update is positive even with a one-dream budget.

Canonical rule for this probe family:

1. Learn next-state and reward prediction separately.
2. Use the world model for dream-state validity and diagnostics.
3. Use a calibrated scalar reward model for imagined reward targets.
4. Spend at most one dream per real step by default; two dreams are better here
   but double compute and should remain an explicit high-budget setting.
5. Do not promote model-based updates unless the same setting beats real-only
   on mean and wins every seed in the 30-seed probe.

## 2026-05-09 LeWM-Style Latent Selection Probe

Command:

```bash
python "examples/The Alberta Plan/Step3/lewm_latent_dream_probe.py" \
  --output-dir output/lewm_latent_dream_probe
```

This probe tests the user's proposed split directly:

1. Learn an environment predictor:
   `(z_t, action_t) -> (z_{t+1}, reward_{t+1}, discount_{t+1})`.
2. Learn an agent/behavior predictor: `P(action_t | z_t)`.
3. Build a counterfactual candidate pool from high-surprise/high-utility real
   anchors, query both possible actions, then select a bounded dream budget by
   surprise, predicted utility, and behavior-model confidence.

Results over 10 seeds, 5,000 steps, 256 anchors, and a 64-item dream budget:

- Stream rare-event rate: `0.04518 +/- 0.06173`.
- Replay-selected rare-event rate: `0.70000 +/- 0.48305`.
- Candidate-pool rare-event rate: `0.21133 +/- 0.23567`.
- Selected dream-candidate rare-event rate: `0.50781 +/- 0.51572`.
- Selected candidate rare enrichment: `3.72x +/- 3.23x`.
- Candidate-pool true reward: `0.17702`; selected true reward: `0.49211`.
- Candidate-pool behavior probability: `0.50000`; selected behavior
  probability: `0.77102`.
- Mean behavior-model NLL: `0.20059`.
- Final latent prediction-error EMA: `0.00060`.
- Fixed-latent SIGReg loss: `0.31109`; latent std mean: `0.06059`.

The canonicalized ablation now writes `ablation_results.csv` and compares
search-control variants:

| Candidate selector | Rare enrichment | Selected true reward | Behavior probability |
| --- | ---: | ---: | ---: |
| Random candidates | `0.52x` | `0.18651` | `0.49814` |
| LeWM surprise only | `0.53x` | `0.02774` | `0.22224` |
| Predicted utility only | `3.71x` | `0.49204` | `0.58431` |
| LeWM surprise + utility | `3.71x` | `0.49223` | `0.57880` |
| Agent-world selector | `3.72x` | `0.49211` | `0.77102` |

Interpretation: this supports a narrower and cleaner selection claim. The useful
signal is not raw LeWM surprise by itself; surprise-only performs like random
or worse in this controlled task. The useful dream budget comes from predicted
utility, with surprise retained as a guard/novelty term and behavior confidence
as a plausibility term. In other words: do not train on all imagined transitions,
and do not train on surprising transitions merely because they are surprising.
Train on model-generated transitions that are surprising enough to be
informative, useful enough to matter, and plausible enough under the learned
agent/behavior model.

The SIGReg diagnostic also matters. The fixed tanh random encoder produces a
low-variance latent cloud, so its SIGReg loss remains far from the isotropic
Gaussian target. This is evidence for the next implementation step: replace the
fixed encoder with a trainable encoder and make SIGReg a real batch loss, not
only a diagnostic.

Promotion status: still research evidence, not a Step 4/7/8 learner claim. The
probe selects dream candidates but does not yet apply those candidates to a
SARSA, Horde actor-critic, or average-reward control update. The next
acceptance experiment should wire the selected candidates into the existing
Horde/SARSA item conversion path and compare real-only, replay-only,
all-dreams, surprise-only, and surprise-plus-utility-plus-confidence updates.

Remaining LeWM gaps:

- replace the fixed encoder with a learned encoder;
- implement the Gaussian latent regularizer as a training loss, not only a
  collapse diagnostic;
- add ensemble or distributional uncertainty before multi-step rollouts;
- test pixels or richer observations after the low-dimensional mechanism earns
  promotion;
- verify online return improvement, not just selected-candidate enrichment.

## Canonical Ablation Matrix

The next comparison should be deliberately small and publishable. Use one
common stream interface and report the same metrics across all rows:

1. Real-only learner.
2. Real + uniform replay from recent real transitions.
3. Real + all model dreams, capped to the same update budget.
4. Real + LeWM surprise-only dreams.
5. Real + predicted-utility dreams.
6. Real + surprise-plus-utility dreams.
7. Real + surprise-plus-utility-plus-behavior-confidence dreams.
8. Real + oracle dreams, as an upper bound on model bias.

Primary tasks:

- common low-dimensional control: CartPole/Catch via the bsuite runner;
- continuous-control sanity: Pendulum or the existing continuous actor-critic
  preview;
- OPMNIST identification: use the Step 2 OPMNIST stream as a continual visual
  identification benchmark, with dreams applied to representation/feature
  learning rather than direct control;
- mixed agent task: alternate control episodes with OPMNIST batches so the
  same feature stack must support action-relevant dynamics and visual identity.

Primary metrics:

- online return or bsuite score for control;
- OPMNIST online accuracy, final-window accuracy, and retention over old
  permutations;
- world-model prediction error and surprise calibration;
- SIGReg loss, latent std mean, and latent projected std;
- dream acceptance/selection rate;
- compute budget in updates per real step and wall-clock throughput.

Promotion rule:

Do not promote "dreaming" generically. Promote only the smallest selector that
beats real-only and replay-only on mean, wins most seeds, and does not degrade
OPMNIST retention. The likely candidate after the current probe is utility-first
selection with surprise and behavior confidence as gates, not surprise-only
LeWM.

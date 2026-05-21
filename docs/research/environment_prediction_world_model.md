# Environment Prediction World Model

Status: implemented observed-state and fixed-latent primitives, 2026-05-09.

## Question

The current Step 3/4 path learns value-like predictions and action values, but
it does not yet learn a reusable model of the environment. The missing piece is
not another control head. It is a temporally uniform predictor of what the
environment will do next:

`(observation_t, action_t) -> (reward_{t+1}, observation_{t+1})`.

That model is the smallest bridge from Step 3 prediction knowledge toward Step
7 planning and Step 8 one-step model-based reinforcement learning.

## Local Audit

Existing repo support before this worker lane:

- `src/alberta_framework/streams/gymnasium.py` can collect supervised reward
  or next-state targets from Gymnasium trajectories. It is data-preparation
  code, not a reusable learned environment model.
- `src/alberta_framework/core/horde.py` and
  `src/alberta_framework/core/off_policy_horde.py` learn GVF cumulants and
  temporally extended predictions. They can represent predictive knowledge, but
  they do not expose a single transition/reward simulator conditioned on a
  candidate action.
- `src/alberta_framework/pipeline.py` accepts caller-supplied Step 3 cumulants
  and can update control plus prediction components per step. It has no learned
  transition model that can be queried for imagined next observations.
- `src/alberta_framework/utils/nexting.py` computes forward-view returns for
  evaluation. It does not learn a model online.

Worker E adds the missing reusable layer in
`src/alberta_framework/core/world_model.py`. The first compatibility surface is
`OneStepWorldModel`; the promoted environment-prediction surface for dreaming is
now `ActionConditionedWorldModel`. The LeWM-inspired latent lane adds
`LatentWorldModel` in `src/alberta_framework/core/latent_world_model.py`.

## Implemented Primitive

`OneStepWorldModel` wraps the existing `MultiHeadMLPLearner`.

Input:

`x_t = concat(observation_t, action_features_t)`

Output heads:

`[reward_hat_{t+1}, observation_hat_{t+1,0}, ..., observation_hat_{t+1,n}]`

For discrete actions, `action_features_t` is a one-hot vector. For continuous
or externally encoded actions, it is a flattened action vector. Missing targets
are represented with `NaN`, reusing `MultiHeadMLPLearner`'s per-head target
masking. The update is supervised and one-step:

`theta_{t+1} = theta_t + update(theta_t, x_t, [r_{t+1}, o_{t+1}])`

There is also an optional `predict_delta` mode where the observation heads
predict `observation_{t+1} - observation_t` and decoded predictions add the
delta back to the current observation.

The production Step 8 facade in `src/alberta_framework/steps/step8.py` exposes:

- `Step8WorldModelConfig`
- `make_step8_world_model`
- `init_step8_state`
- `step8_update`
- `run_step8_scan`
- `run_step8_smoke`

## Action-Conditioned Discount Extension

The current dream-ready model is `ActionConditionedWorldModel`. It keeps the
same multi-head learner substrate, but predicts a fuller transition object:

`(observation_t, action_t) -> (delta_observation_{t+1}, reward_{t+1}, discount_{t+1})`

The discount head is important because self-simulation needs an explicit
continuation signal. For episodic Gymnasium tasks the empirical target is
`gamma` on continuing steps and `0` on terminal/truncated steps. For continuing
tasks it can remain near the configured `gamma` or become a learned
pseudo-termination question.

The model also tracks observed observation/reward bounds and a real-transition
model-error EMA. Those diagnostics are intentionally modest: they are enough to
gate early Dyna-style rollouts, but they are not calibrated uncertainty.

Implemented API:

- `ActionConditionedWorldModelConfig`
- `ActionConditionedWorldModel`
- `ActionConditionedWorldModelState`
- `ActionConditionedWorldModelLearningResult`
- `run_action_conditioned_world_model_learning_loop`
- `WorldModelPrediction` with `next_observation`, `reward`, `discount`, and
  raw heads

The implementation is exported from both `alberta_framework.core` and
`alberta_framework`.

## Fixed-Latent LeWM-Inspired Extension

`LatentWorldModel` is the first local bridge from one-step observed-state
prediction toward JEPA-style world modeling. It learns:

`(z_t, action_t) -> (delta_z_{t+1}, reward_{t+1}, discount_{t+1})`

where `z_t = tanh(encoder(observation_t))`. The encoder is fixed in this first
version, so this is not a full representation-learning result. The point is to
test the Alberta-compatible mechanics before moving to pixels or learned
encoders:

- immutable JAX state;
- scan-compatible online update loop;
- per-transition latent surprise as prediction error;
- reward and discount heads for Dyna/SARSA/Horde conversion;
- optional latent-by-action interaction features;
- latent variance/collapse diagnostics inspired by JEPA anti-collapse needs;
- reusable SIGReg loss/diagnostics in `src/alberta_framework/core/sigreg.py`;
- config roundtrip and package exports.

The current `LatentWorldModel` deliberately does not train the encoder with a
Gaussian regularizer. The repo now has the full differentiable sliced
Epps-Pulley/BHEP SIGReg primitive, but the fixed-encoder model only uses SIGReg
as a diagnostic. A future trainable encoder should optimize:

`next_embedding_prediction_loss + lambda * SIGReg(latents)`

with gradients through both encoder and predictor, matching the LeWM training
logic more closely.

## Empirical Harness

The first runnable environment-prediction benchmark is:

```bash
python "examples/The Alberta Plan/Step3/action_conditioned_cartpole_dynamics.py"
```

It runs random-policy `CartPole-v1` transitions and compares:

- `linear`: no hidden trunk, action-conditioned delta/reward/discount heads;
- `mlp`: one hidden layer with the same transition targets.

Outputs:

- `output/action_conditioned_cartpole/results.csv`
- `output/action_conditioned_cartpole/summary.json`

Primary metrics:

- prequential next-observation RMSE;
- final-window next-observation RMSE;
- ratio against a zero-delta baseline;
- reward and discount RMSE;
- final guarded-dream acceptance.

Current local evidence:

- Command: `python "examples/The Alberta Plan/Step3/action_conditioned_cartpole_dynamics.py" --output-dir output/action_conditioned_cartpole`
- Protocol: 10 seeds, 20,000 random-policy `CartPole-v1` transitions per seed,
  2,000-step final window.
- Result path: `output/action_conditioned_cartpole/{results.csv,summary.json}`.
- Linear final-window next-observation RMSE: `0.00845 +/- 0.00259`.
- MLP final-window next-observation RMSE: `0.01363 +/- 0.00132`.
- Linear ratio vs zero-delta baseline: `0.04799 +/- 0.01475`.
- MLP ratio vs zero-delta baseline: `0.07739 +/- 0.00746`.
- MLP minus linear final-window next-observation RMSE: `0.00518`; MLP better
  seeds: `0/10`.
- Dream acceptance after training: `1.0` for both methods under the permissive
  guard used by this dynamics-only benchmark.

Interpretation: on CartPole dynamics under random behavior, the simple linear
delta model is the best current candidate for one-step environment prediction.
The MLP condition learns a lower final-window discount RMSE (`0.18542` vs
`0.21153`) but gives worse next-observation prediction, so it should not be the
default Dyna substrate without retuning.

## Why This Makes Sense

The user hypothesis is right: a control learner that only updates action values
is reactive. It can learn which action currently works, but it does not learn
what the action does to the world. A learned environment predictor adds a second
causal object:

- behavior learner: predicts or chooses the agent's action;
- environment learner: predicts the world's next observation and reward under
  that action.

That split matters because planning, self-simulation, and counterfactual action
evaluation require a model that can answer "what if I took action `a` here?"
without executing `a` in the real environment.

## Relation to Alberta Plan Steps

Step 3: GVFs and Horde

GVFs ask temporally extended predictive questions grounded in sensorimotor
experience. A one-step world model is not a replacement for GVFs. It is a
different predictive object: a direct transition/reward model that can provide
candidate imagined observations and rewards. GVFs can then evaluate those
imagined states, or the world-model prediction errors can become cumulants for
feature discovery.

Step 7: Planning

Step 7 calls for incremental planning in continuing, average-reward settings.
This lane does not implement planning. It provides the model interface that a
planning worker can query. The next planning layer should decide how many
background backups are allowed per real step and how to choose states/actions
for hypothetical updates.

Step 8: One-step model-based RL

The implemented model is the Step 8 substrate: a learned one-step model with
continual function approximation. It can be used by Dyna-style real/hypothetical
updates or by a small rollout/dream adapter, but those are intentionally left
for the planning/dreaming lane.

## Literature Placement

Known work:

- Dyna integrates reinforcement learning, online model learning, and planning
  over a learned forward model. Sutton's Dyna paper describes architectures
  that learn a world model online while using dynamic-programming-style planning
  to improve behavior ([Sutton 1990](https://papers.nips.cc/paper/1990/file/d9fc5b73a8d78fad3d6dffe419384e70-Paper.pdf)).
- Predictive state representations frame state as action-conditional
  predictions of future observations, rather than latent variables assumed by
  the designer. Littman, Sutton, and Singh argue that dynamical-system state can
  be represented by multi-step, action-conditional future-observation
  predictions ([NeurIPS 2001](https://papers.neurips.cc/paper/1983-predictive-representations-of-state)).
- Horde/GVFs express sensorimotor knowledge as many learned value-function
  predictions and show real-time scalability with many demons
  ([Sutton et al. 2011](https://sites.ualberta.ca/~pilarski/docs/papers/Sutton_2011_Horde_AAMAS.pdf)).
- Ha and Schmidhuber's World Models learn compact environment representations
  and can train policies inside generated rollouts
  ([arXiv:1803.10122](https://arxiv.org/abs/1803.10122)).
- PlaNet and Dreamer move this idea into learned latent dynamics and
  imagination-based behavior learning from pixels
  ([PlaNet](https://arxiv.org/abs/1811.04551),
  [Dreamer](https://arxiv.org/abs/1912.01603)).
- LeWorldModel is the closest current reference for the user's "predict the
  world before acting on it" hypothesis. It is an action-conditioned
  JEPA-style model trained from pixels with next-embedding prediction plus a
  Gaussian latent regularizer, and it uses latent prediction error as surprise
  ([Maes et al. 2026](https://arxiv.org/abs/2603.19312)).
- DreamerV3 and TD-MPC2 show modern world-model systems scaling across many
  tasks with fixed or broad hyperparameter settings
  ([DreamerV3](https://arxiv.org/abs/2301.04104),
  [TD-MPC2](https://arxiv.org/abs/2310.16828)).

Novel in this repo:

- The implementation is deliberately small and temporally uniform: one update
  per real transition, no replay buffer, no special offline phase, and no
  planning side effects.
- It reuses the Step 2/3 multi-head learner and therefore inherits the same
  optimizer/bounder/normalizer surface as the rest of the framework.
- It makes the transition model an ordinary continual learner that can be
  inspected with the same per-head metrics and hidden-utility diagnostics.

Not novel:

- One-step next-state and reward prediction is a standard model-based RL
  primitive.
- Action conditioning, one-hot action features, and supervised transition-model
  learning are standard.
- This is not a new Dreamer/PlaNet-style latent dynamics architecture.

## Working Memory and Dreaming Interface

This lane now implements the narrow self-simulation interface those components
need, without promoting a planning learner yet.

Candidate next pieces:

- Working memory: maintain a recurrent or finite-history feature state `m_t`
  and train the world model on `concat(o_t, m_t, a_t)`. Existing
  `HistoryFeatureExtractor` and temporal-context modules are the nearest local
  starting points.
- Behavior model: learn `pi_hat(a_t | o_t, m_t)` or a deterministic behavior
  predictor from the same features. This is separate from the environment
  model, and it matters for off-policy imagined rollouts that should resemble
  observed behavior.
- Dreaming/self-simulation: seed imagined rollouts with a real observation or
  memory state, sample or choose actions from the behavior/control model, query
  `ActionConditionedWorldModel.predict`, and feed imagined transitions into
  value/GVF updates with strict budgets and model-error gating.

Local implementation:

- `src/alberta_framework/core/dreaming.py` defines `GuardedDreamer` for
  one-step proposals from `ActionConditionedWorldModel`.
- `RecentObservationBuffer` anchors dreams in recently observed real states.
- `DreamWorldModel` and `DreamBehaviorModel` protocols separate environment
  prediction from behavior/action prediction.
- `ActionConditionedDreamWorld` adapts the concrete world model to short
  rollouts.
- `dream_rollout`, `imagined_rollout_to_gvf_items`, and
  `imagined_rollout_to_sarsa_items` convert imagined transitions back into the
  same learner item shapes used by real transitions.
- `DreamSelectionConfig` ranks candidate imagined transitions by surprise,
  utility, behavior-model confidence, and model-error penalties.

Acceptance criteria for the dreaming lane should include model-error
diagnostics and a no-regression test where imagined updates are disabled when
prediction error is high.

## Remaining Blockers

- No stochastic uncertainty head or ensemble disagreement yet. The model
  predicts point estimates and only exposes real-transition error EMA.
- No learned latent encoder yet. `LatentWorldModel` has fixed random features
  plus collapse diagnostics, not a trainable LeWM/SIGReg representation.
- No recurrent latent state. Partially observable environments still require
  external history/working-memory features.
- No planning/search-control policy. The model can be queried but does not
  decide which imagined updates to run.
- A behavior model exists as a separate component, but it is not yet wired into
  a production Dyna control loop.
- No empirical proof yet that model-based updates improve Step 4 control in
  this repo. The CartPole dynamics sweep only tests environment prediction.

## 2026-05-07 Rerun

Command:

```bash
python "examples/The Alberta Plan/Step3/action_conditioned_cartpole_dynamics.py" \
  --output-dir output/action_conditioned_cartpole_rerun \
  --train-steps 20000 \
  --seeds 10 \
  --final-window 2000
```

Result: the earlier CartPole dynamics result reproduced. The linear
action-conditioned delta model remains the best current one-step dynamics
candidate:

- Linear final-window next-observation RMSE: `0.00845`.
- MLP final-window next-observation RMSE: `0.01363`.
- MLP minus linear RMSE: `+0.00518`.
- MLP better seeds: `0/10`.
- Dream acceptance after training: `1.0` for both methods under the permissive
  dynamics-only guard.

Interpretation: for low-dimensional CartPole dynamics, linear delta prediction
is currently superior to the tested MLP substrate. Dreaming/control experiments
should start from the linear model, not the MLP branch.

## 2026-05-09 Latent Surprise Selection Probe

Command:

```bash
python "examples/The Alberta Plan/Step3/lewm_latent_dream_probe.py" \
  --output-dir output/lewm_latent_dream_probe
```

This synthetic probe tests whether a fixed-latent environment model plus a
behavior model can select useful imagined counterfactuals rather than training
on all generated model transitions. It uses rare useful jumps, trains
`LatentWorldModel` and `BehaviorModel` online, then scores candidate dreams by
latent surprise, predicted reward, and behavior-model confidence.

Canonical 10-seed result:

- Candidate-pool rare-event rate: `0.21133`.
- Selected dream-candidate rare-event rate: `0.50781`.
- Selected candidate rare enrichment: `3.72x`.
- Candidate-pool true reward: `0.17702`.
- Selected true reward: `0.49211`.
- Selected behavior probability: `0.77102` versus `0.50000` in the full
  candidate pool.
- Final latent prediction-error EMA: `0.00060`.
- Fixed-latent SIGReg loss: `0.31109`, with latent std mean `0.06059`.

Selector ablation:

- Random candidates: `0.52x` rare enrichment.
- LeWM surprise-only: `0.53x` rare enrichment.
- Predicted utility-only: `3.71x` rare enrichment.
- LeWM surprise + utility: `3.71x` rare enrichment.
- Agent-world selector: `3.72x` rare enrichment and higher behavior
  probability (`0.77102`).

Interpretation: the environment-prediction path now has a concrete selection
mechanism for self-simulation. It supports the claim that the agent should not
consume all world-model samples uniformly, but it rejects a raw
"surprise-only" reading of LeWM. In this probe, predicted utility is the main
selector, surprise is best treated as novelty/gating information, and behavior
confidence improves plausibility. The next step is to use the selected dream
candidates as actual Horde/SARSA/actor-critic updates and prove that online
return or prediction error improves against real-only and replay-only controls.

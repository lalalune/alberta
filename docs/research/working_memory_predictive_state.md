# Working Memory Predictive State Track

Date: 2026-05-07

## Decision

Added `core.working_memory` as a small, composable predictive-state feature
module rather than extending `history_features.py` or `temporal_context.py`.

The audit result was:

- `history_features.py` already supplies observation-only multi-timescale
  traces. It is useful, but it cannot represent action-conditioned history or
  reward/cumulant history.
- `temporal_context.py` supplies Step 2 causal EMA/delta/phase features for
  non-stationary supervised streams. It is intentionally context-feature
  oriented, not a general agent memory.
- `prototype_memory.py` and `associative_memory.py` are retained-view memories
  for classification-style recall. They are not the right substrate for a
  low-level transition model because they allocate class prototypes.

The new module keeps exponentially decayed traces over observations, actions,
and rewards:

```text
m_t = m_{t-1} + g_t (1 - beta) (x_t - m_{t-1})
```

where `g_t = 1` by default and can be supplied externally or computed from a
simple surprise gate. Features are emitted before the current transition is
written into the trace banks, with optional current observation/action/reward
blocks included. This allows a caller to learn:

- environment prediction: `o_{t+1}, r_{t+1} <- f(memory_t, o_t, a_t)`;
- behavior prediction: `a_t <- f(memory_t, o_t)`;
- Horde cumulant/termination predictions from memory features;
- actor-critic policies and critics under partial observability.

## Why Environment Prediction Makes Sense Here

The current Step 3/4 pipeline mostly learns values or GVFs from features and
then updates control from actual environment transitions. That is legitimate
for Steps 3/4, but it leaves the Step 7/8 bridge underspecified: the agent has
no compact, shared state that says "given what I have observed and done, what
environment am I in, and what would happen next if I acted differently?"

A predictive-state memory is the smallest missing bridge. It does not require
offline replay, a special training phase, or a large recurrent model. Every
real transition updates the memory, every model can consume the same memory
vector, and later Dyna/dreaming can reuse the model's own predictions as
simulated observation/action/reward transitions.

The immediate architecture is:

```text
experience_t = (observation_t, action_t, reward_t)
        |
        v
WorkingMemoryFeaturizer
        |
        +--> environment model: predict observation_{t+1}, reward_{t+1}
        +--> behavior model: predict action_t or action_{t+1}
        +--> Horde/GVFs: predict many future cumulants
        +--> actor-critic: condition policy/value on memory
```

The dreaming/self-simulation track should then add:

1. a one-step environment model trained online from working-memory features;
2. a behavior model or policy prior trained online from the same features;
3. a short-horizon simulator that rolls memory forward using predicted
   observations/rewards and sampled or policy-proposed actions;
4. Dyna-style planning updates that are budgeted per real step, not run as a
   separate offline phase.

## Known Literature

- The Alberta Plan frames intelligence around continual experience and says
  the transition model should predict next state and reward from state/action
  while learning and planning remain temporally uniform
  ([Sutton, Bowling, and Pilarski, 2023](https://arxiv.org/abs/2208.11173)).
- Predictive state representations replace latent-state hypotheses with
  action-conditional predictions of future observations; Littman and Sutton
  show this can be a compact state representation for controlled dynamical
  systems
  ([Predictive Representations of State](https://papers.neurips.cc/paper/1983-predictive-representations-of-state)).
- TD networks generalize TD learning to networks of predictions and explicitly
  connect predictive representations to non-Markov problems
  ([Sutton and Tanner, 2004](https://papers.neurips.cc/paper/2545-temporal-difference-networks)).
- Gradient TD networks show why naive inter-predictive networks can diverge and
  motivate stable gradient-TD machinery for future nonlinear predictive-state
  work
  ([Silver, 2013](https://proceedings.mlr.press/v24/silver12a.html)).
- Nexting/Horde demonstrates real-time learning of thousands of GVF predictions
  over sensory signals and timescales on a robot
  ([Modayil, White, and Sutton, 2014](https://journals.sagepub.com/doi/abs/10.1177/1059712313511648)).
- Dyna is the direct model-learning/planning precedent for using learned models
  to create simulated experience
  ([Sutton, 1990/1991](https://papersdb.cs.ualberta.ca/~papersdb/view_publication.php?pub_id=505)).
- World Models, Dreamer, and MuZero show stronger modern versions of
  environment prediction and latent imagination, but they are not temporally
  uniform Alberta-Plan baselines as usually implemented because they rely on
  replay/offline training phases, large batch optimization, or search budgets
  outside the simple per-step foreground loop
  ([Ha and Schmidhuber, 2018](https://arxiv.org/abs/1803.10122),
  [Hafner et al., 2019](https://arxiv.org/abs/1912.01603),
  [Schrittwieser et al., 2019](https://arxiv.org/abs/1911.08265)).
- Successor features are another predictive representation of environment
  dynamics decoupled from reward specification
  ([Barreto et al., 2017](https://papers.nips.cc/paper/6994-successor-features-for-transfer-in-reinforcement-learning)).
- Recent memory-trace work gives direct support for exponential traces as a
  compact alternative to long finite histories in partially observable RL
  ([Eberhard, Muehlebach, and Vernade, 2025](https://arxiv.org/abs/2503.15200)).

## What Is Novel Locally

The local contribution is not the idea of traces, predictive states, or world
models. Those are established. The local contribution is a package-level,
JAX/scan-compatible bridge that:

- tracks observation, action, and reward history in one temporally uniform
  state object;
- emits a single feature vector usable by environment prediction, behavior
  prediction, Horde, and actor-critic;
- supports optional causal gates without changing the public state contract;
- exposes diagnostics for energy, effective dimension, and last update gates;
- avoids an LSTM/replay dependency while leaving room for learned nonlinear
  models to sit on top.

## Acceptance Status

Accepted as infrastructure, not as Step 7/8 closure.

The module passes focused tests for finite updates, reset semantics, trace
decay, action/reward inclusion, scan compatibility, diagnostics, and a delayed
action positive control. It does not yet prove that world-model dreaming helps
control, nor does it choose the best memory timescales automatically.

## Remaining Blockers

- Implement a one-step environment model:
  `memory_features_t, action_t -> observation_{t+1}, reward_{t+1}`.
- Implement a behavior model:
  `memory_features_t, observation_t -> action_t` or the policy's next action.
- Add a self-simulation loop that rolls `WorkingMemoryFeaturizer.update`
  forward on predicted transitions.
- Budget model/planning updates per real step so dreaming remains temporally
  uniform.
- Test on partial-observation and delayed-target tasks before using this as a
  default Step 3/4 input.

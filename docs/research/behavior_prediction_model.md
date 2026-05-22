# Behavior Prediction Model

Worker F adds a small temporally uniform behavior model:

```python
BehaviorModel(BehaviorModelConfig(n_actions=k, ...))
```

It learns `P(A_t | x_t)` online from the actually executed discrete action. The
input `x_t` is whatever feature vector the current agent already uses: raw
observation features, Step 2 features, history features, Horde features, or a
future working-memory state. This is an action model, not a control policy. It
does not choose what the real agent should do unless a caller explicitly samples
from it for diagnostics or dreamed rollouts.

## Why This Makes Sense

The current Step 3/4 stack mostly updates the agent from `S_t, A_t, R_{t+1},
S_{t+1}`. That is enough for SARSA and one-step Horde predictions, but it does
not give the architecture a persistent estimate of either:

- the world transition distribution `P(S_{t+1}, R_{t+1} | S_t, A_t)`;
- the behavior distribution `P(A_t | S_t, history_t)` that generated data.

For off-policy learning, the second quantity is not optional. Importance ratios
need a behavior denominator. For self-simulation, the first quantity predicts
what the environment will do, while the second predicts what this agent, a stale
policy, a human, or a scripted behavior source would do in the simulated state.

This addition closes only the behavior side. The companion world-model lane
should learn next-observation/reward/discount predictions.

## Existing Policy Surfaces

`SARSAAgent.select_action` uses epsilon-greedy selection over Horde Q-heads:
exploration is uniform over all actions; exploitation is greedy with random
tie-breaking. The reusable exact distribution is now exposed as
`epsilon_greedy_probabilities(q_values, epsilon)`.

`ActorCriticAgent` and `HordeActorCriticAgent` use linear softmax actors:
`softmax((W x + b) / temperature)`, then sample with `jax.random.categorical`.
The behavior model uses the same probability family, but its objective is
supervised cross-entropy on observed actions rather than policy-gradient control.

This gives a clean API:

- `predict_logits(state, x)` for raw action scores;
- `predict_probabilities(state, x)` for the behavior distribution;
- `update(state, x, action)` every real step;
- `action_probability` and `action_log_likelihood` for diagnostics;
- `importance_ratio(state, x, action, target_probs)` for off-policy consumers;
- `sample_action(state, x)` for imagined behavior in dream rollouts.

## Known Research Context

The Alberta Plan emphasizes continual, temporally uniform components that update
from ordinary experience; a behavior predictor fits that discipline because it
learns every time step from the same experience stream as control and prediction
[Sutton, Bowling, and Pilarski 2023](https://arxiv.org/abs/2208.11173).

Dyna integrated acting, model learning, and planning around a learned forward
model of the world [Sutton 1990](https://papers.nips.cc/paper_files/paper/1990/hash/d9fc5b73a8d78fad3d6dffe419384e70-Abstract.html).
Modern world-model systems such as World Models and Dreamer use learned latent
dynamics for imagined experience and policy improvement
([Ha and Schmidhuber 2018](https://arxiv.org/abs/1803.10122),
[Hafner et al. 2019](https://arxiv.org/abs/1912.01603)). MBPO shows why short
model rollouts are usually safer than long hallucinations when model error can
compound [Janner et al. 2019](https://arxiv.org/abs/1906.08253).

Behavior cloning is the standard supervised action-model objective: learn a
policy distribution from state-action pairs. Here the same cross-entropy
objective is used, but as an internal model of the behavior stream rather than
as the promoted controller.

Off-policy evaluation/control needs behavior probabilities for importance
sampling. Estimated behavior policies are a known tool when the true behavior
policy is unavailable or nonstationary; Hanna, Niekum, and Stone study this
explicitly in off-policy evaluation
([arXiv 2018](https://arxiv.org/abs/1806.01347),
[Machine Learning 2021](https://link.springer.com/article/10.1007/s10994-020-05938-9)).

## What Is Known vs Novel Here

Known:

- supervised action prediction with cross-entropy;
- epsilon-greedy and softmax policy probability formulas;
- importance-ratio correction using behavior denominators;
- Dyna/Dreamer-style use of learned models for simulated experience.

Local novelty:

- a JAX/scan-compatible behavior model packaged as an Alberta Framework core
component;
- a single API that can estimate SARSA, Horde actor-critic, scripted, external,
or stale-policy behavior without modifying those control agents;
- online reliability diagnostics (`nll_ema`, `accuracy_ema`, `confidence_ema`)
that can gate off-policy ratios or future dream rollouts;
- denominator-safe helpers aligned with `OffPolicyHordeLearner`'s clipped-ratio
backend.

## Reliability Diagnostics

The model returns the current action probability, log-likelihood, loss, entropy,
confidence, predicted action, and correctness. The state maintains EMA negative
log-likelihood, accuracy, and confidence. A future dream planner should treat
high NLL, low action probability, or confidence/accuracy mismatch as a reason to
shorten rollouts, lower dream update weight, or branch dreams from more recent
real states.

## Acceptance Boundary

This lane is now wired to the dreaming surface through
`BehaviorModelDreamPolicy`, but it remains the behavior-policy half of the
proposed `(world model, behavior model)` pair. Observed-state and fixed-latent
environment predictors exist separately; full promotion still requires:

- short-rollout dream updates with model-error and behavior-confidence gates;
- ablations comparing exact known behavior probabilities vs learned behavior
  probabilities in off-policy Horde;
- tests showing that dream updates help on nonstationary tasks without hurting
  real-stream online tracking.

The current LeWM-style latent probe shows that behavior confidence is useful as
a search-control term: selected dream candidates have mean behavior probability
`0.77102` versus `0.50000` in the full candidate pool while also enriching rare
useful transitions by `3.72x`.

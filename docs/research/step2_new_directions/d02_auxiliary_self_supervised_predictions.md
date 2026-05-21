# D02 Temporal Auxiliary Prediction Heads

## Core Hypothesis

A fair Step 2 MLP loses useful features because the supervised target is a
narrow, delayed credit signal. Add online self-supervised auxiliary heads to the
same trunk, trained at every time step to predict random projections of the next
observation. The auxiliary loss should make the trunk encode predictive state
information that is not always immediately rewarded by the supervised heads,
improving adaptation in nonstationary and recurring-context streams without
changing the primary supervised target.

## Why Different

This is not another target-code, feature-lift, reset, ensemble, prototype,
orthogonalization, or parameter-averaging trick. The prior ten moonshots changed
the supervised target geometry, installed fixed or imprinted features, changed
plasticity after surprise, blended predictors, constrained hidden geometry, or
changed the prediction weights. D02 changes the online objective: every tick
produces both a supervised update and a task-free predictive update through the
same learned representation. The mechanism is closer to predictive-state
learning than to static feature construction.

## Mathematical Grounding

For a stream `(x_t, y_t)`, use the standard MLP trunk `h_t = f_theta(x_t)` with
supervised head `y_hat_t = W_y h_t`. Draw one fixed row-normalized Rademacher
projection matrix `R in R^(k x d)` per seed, with `k` much smaller than the
input dimension. Maintain the existing online observation normalizer `n_t`.
When `x_(t+1)` arrives, form a delayed auxiliary target for the previous input:

```text
a_t = R n_t(x_(t+1))
L_t = ||W_y f_theta(x_t) - y_t||_2^2
    + lambda_aux ||W_a f_theta(x_t) - stop_gradient(a_t)||_2^2
```

The one-step delay avoids future-label leakage: at wall-clock step `t+1`, the
learner updates on `(x_t, y_t, x_(t+1))`, then predictions for `x_(t+1)` are
scored before its own supervised update. This preserves temporal uniformity:
the trunk, supervised head, auxiliary head, normalizer, optimizer state, and
bounder all update once per stream tick after the initial buffer fill.

Random projection keeps the auxiliary head small while approximately preserving
next-observation geometry. Predicting `R x_(t+1)` instead of full `x_(t+1)`
prevents the experiment from becoming a large autoencoder with an unfair output
budget. The auxiliary target is exogenous, so all gradients into the trunk come
from whether the current representation predicts future sensory structure.

## Why It Could Beat The Previous Iteration And Fair MLP

The fair MLP only receives gradients for variables that are currently useful
for the supervised target. In Step 2 streams with changing contexts, hidden
latents, cyclic phases, or delayed relevance, that can delete or undertrain
features before they become useful again. The auxiliary prediction head supplies
dense gradient on every step for features that explain temporal continuity,
scale drift, recurring contexts, and latent dynamics. If useful supervised
features are also predictive of near-future inputs, D02 should discover them
earlier and retain them longer.

This could improve over the previous moonshot batch because the signal is not
hand-coded for a particular output encoding or interaction family. It should
help most when the stream has real temporal structure and should not help when
time is shuffled. That gives the experiment a built-in falsification test.

## Minimal Implementation Sketch

Create an experiment-local wrapper around `MultiHeadMLPLearner`, not a shared
core API change. The wrapper concatenates primary and auxiliary heads:
`n_heads = y_dim + k`. For auxiliary weighting, scale the auxiliary targets by
`sqrt(lambda_aux)` and report metrics only on the first `y_dim` heads. Use the
same trunk width, sparse initialization, layer norm setting, ObGD bounding, and
optimizer grid as the fair MLP baseline.

Protocol:

- Baselines: fair supervised MLP, wider fair MLP if parameter count is disputed,
  and D02 with `lambda_aux in {0.03, 0.1, 0.3}` and `k in {8, 16, 32}`.
- Streams: the existing synthetic out-of-class suite, plus shuffled digits,
  class-blocked digits, and at least one time-shuffled version of each
  temporally structured stream.
- Evaluation order: prequential score first, delayed update second, identical
  for baseline and D02.
- Keep all code in the future experiment script unless the wrapper proves
  generally reusable.

## Metrics And Success Bar

Primary metrics are final-window supervised MSE on synthetic streams and
final-window plus held-out accuracy/MSE on digits. Report paired seed
differences against the best fair MLP, not only means. Secondary diagnostics are
auxiliary prediction MSE/R2, feature relevance, feature sensitivity, and
normalizer lag.

Pilot success: with 10 paired seeds and 3000-6000 online steps, at least one D02
configuration must beat the best fair MLP by `>= 5%` final-window supervised MSE
on two temporally structured streams, with at least `8/10` paired wins on each,
while losing no more than `2%` relative accuracy or MSE on shuffled digits.
Promotion success: repeat the winning configuration at 30 seeds and require the
effect to survive the time-shuffled negative control.

## Risks And Negative Controls

The auxiliary target can become pure noise on IID streams, dominate the primary
loss, or reward low-level input persistence rather than useful supervised
features. A flawed protocol can also leak future information if `x_(t+1)` is
used before prequential scoring.

Required controls:

- `lambda_aux = 0`, which must reproduce the fair MLP wrapper.
- Time-shuffled `x_(t+1)` targets; gains here imply regularization or capacity,
  not temporal prediction.
- Fresh random auxiliary targets with the same variance as `R x_(t+1)`.
- Aux-head-only gradient stop into the trunk; this tests whether extra head
  parameters alone explain the result.
- IID shuffled digits, where next-observation prediction should not be a large
  advantage.

## Exact First Command

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d02_auxiliary_self_supervised_predictions.py" --output-dir outputs/step2_new_directions/d02_auxiliary_self_supervised_predictions --seeds 10 --num-steps 3000
```

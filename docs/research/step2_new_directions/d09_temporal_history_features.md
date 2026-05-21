# D09 Causal History-State Features

## Core Hypothesis

Step 2 fair MLPs are penalized on partially observable and nonstationary streams
because they receive only the instantaneous observation `x_t`. A small causal
history-state feature wrapper, built from eligibility-like input traces and
delay embeddings, should expose latent phase, regime, velocity, and recent
context without changing the supervised target or adding a recurrent learner.
The MLP can then solve a Markovized supervised problem with the same optimizer,
bounder, trunk size, and temporal update schedule.

## Why Different

This is not centered targets, ECOC, RFF, hashed quadratic features, residual
imprinting, a fast/slow ensemble, surprise reset, prototype context, hidden
orthogonalization, or EMA prediction. It also differs from auxiliary predictive
heads: D09 adds no extra losses and makes no future-observation target. The only
change is the information state supplied to the existing supervised learner. It
tests whether prior Step 2 failures were due to missing temporal state rather
than inadequate nonlinear approximation at a single time slice.

## Mathematical Grounding

For a stream `(x_t, y_t)` with hidden state `s_t`, the observation may be
non-Markov: `P(y_t | x_t)` can mix several regimes that are separable under
`P(y_t | x_t, x_(t-1), ..., x_(t-L))`. Takens-style delay embeddings motivate
using recent observations to reconstruct latent state when dynamics are smooth;
predictive-state and fading-memory results motivate stable linear filters when
exact finite lags are too brittle.

Maintain a fixed causal feature state before the supervised update:

```text
q_t = normalize(x_t)
e_t^(k) = rho_k e_(t-1)^(k) + (1 - rho_k) q_t
d_t^(k) = e_t^(k) - e_(t-1)^(k)
ell_t = [q_t, q_(t-1), q_(t-2), q_(t-4), q_(t-8), e_t^(1:K), d_t^(1:K)]
```

with `rho_k` logarithmically spaced, for example `{0.25, 0.5, 0.75, 0.9,
0.97}`. `ell_t` is a causal sufficient-statistic approximation: finite lags
capture short delays and alias-breaking details, while leaky traces provide a
bounded-dimensional basis over longer histories. The derivative traces `d_t`
act like velocity or surprise-in-history features without resetting anything.
All trace states update every time step, including when the supervised loss is
large, small, or temporarily uninformative.

To keep parameter count fair, project the raw history vector with a fixed
row-normalized sparse Johnson-Lindenstrauss matrix `P`:

```text
z_t = [q_t, P ell_t]
y_hat_t = MLP_theta(z_t)
```

The baseline MLP still receives `q_t`; D09 receives `q_t` plus a fixed-width
causal memory sketch. No gradients flow into `P` or the trace dynamics.

## Why It Could Beat Previous Iteration And Fair MLP

The fair MLP can only infer regime from instantaneous features. In streams with
class blocks, cyclic phases, scale drift, delayed labels, sensor aliasing, or
hidden switching dynamics, the same `x_t` may require different predictions
depending on recent history. Extra hidden width cannot reliably fix that if the
distinguishing information is absent from the input.

D09 could beat the previous iteration because it attacks a different bottleneck:
state observability. Prior moonshots mostly changed target geometry, static
features, parameter dynamics, or feature maintenance. If supervised errors are
caused by aliasing, a causal history sketch should reduce irreducible error
before optimization matters. Against a fair parameter-matched MLP, D09 should
adapt faster after regime boundaries because trace features expose change
direction and recent context immediately, rather than requiring the trunk to
memorize temporal structure in weights.

## Minimal Implementation Sketch

Implement as an experiment-local wrapper, not a core API change. Add
`HistoryFeatureState` containing the lag ring buffer, multi-timescale traces,
previous traces for derivatives, normalizer state, and fixed sparse projection.
On every stream tick: normalize `x_t`, update lags and traces, build `z_t`,
score prequentially, then call the unchanged `MultiHeadMLPLearner.update()` on
`z_t` and `y_t`.

Compare four learners under paired seeds and identical streams:

- Fair MLP on `q_t`.
- Parameter-matched fair MLP with enough hidden units to match D09 parameter
  count.
- D09 finite lags only: `[q_t, q_(t-1), q_(t-2), q_(t-4), q_(t-8)]`.
- D09 lags plus leaky trace sketch with `history_dim in {16, 32, 64}`.

Primary streams should include existing nonstationary synthetic streams plus at
least two deliberately partially observable variants: delayed-target regression
`y_t = f(x_(t-L))`, and hidden-regime switching where two regimes share the same
marginal `x_t` but differ in the mapping to `y_t`. Include sklearn digits
variants only as external controls; this mechanism should not be tuned around
IID image classification.

## Metrics / Success Bar

Primary metrics: final-window supervised MSE, whole-stream prequential MSE,
paired seed wins, and adaptation half-life after regime switches. Secondary
metrics: performance by delay length, trace-feature ablations, update bound
scale, feature relevance assigned to lag and trace blocks, and wall-clock
overhead.

Pilot success requires D09 lags plus traces to beat the best fair MLP by at
least `8%` final-window MSE on two partially observable or delayed streams,
with at least `8/10` paired wins on each, while the parameter-matched fair MLP
does not close more than half the gap. Promotion requires 30 paired seeds, no
more than `2%` aggregate regression loss on fully observed control streams, and
no statistically meaningful gain on time-shuffled-history controls.

## Risks / Negative Controls

Risks: history features can increase input dimension, slow adaptation when old
context is misleading, leak evaluation information if lags are updated in the
wrong order, or simply regularize the MLP through a larger fixed projection.
They may also hurt abruptly switching streams if slow traces dominate recent
evidence.

Required negative controls:

- Time-shuffled history buffer with the same marginal feature distribution.
- Trace-only features with `x_t` removed, which should fail on fully observed
  instantaneous tasks.
- Current-input duplicated to match D09 dimension, testing capacity and scaling.
- Random Gaussian history sketch independent of past inputs.
- Fully observed IID stream where causal history should not materially help.
- Anti-causal bug check: update history after prequential scoring and verify no
  future `x_(t+1)` can enter `z_t`.

## Exact First Command

After adding the standalone experiment script, run:

```bash
source .venv/bin/activate && PYTHONPATH=src python "examples/The Alberta Plan/Step2/new_directions/d09_temporal_history_features.py" --num-steps 3000 --seeds 10 --history-dim 32 --lags 1 2 4 8 --trace-rhos 0.25 0.5 0.75 0.9 0.97 --streams delayed_regression hidden_regime_switch cyclic_stream scale_drift --output-dir outputs/step2_new_directions/d09_temporal_history_features
```

# D04 Adaptive Spline Basis With Online Knot Splitting

## Core Hypothesis

A bounded bank of adaptive piecewise-linear spline features can beat a fair MLP
in Step 2 streams where the target has localized nonlinear structure, changing
regimes, or sharp slope changes. The key bet is that online feature discovery
should sometimes allocate resolution in input regions with persistent residual
error, instead of relearning a dense hidden representation everywhere.

## Why Different

This is not centered targets, ECOC, random Fourier features, hashed quadratic
features, residual imprinting, an ensemble, surprise reset, prototype context,
hidden orthogonalization, or EMA prediction. It changes the representation into
an adaptive local basis. Unlike RFF, the basis is not frozen and global; unlike
hashed quadratic lift, it does not enumerate monomial interactions. Its unit of
adaptation is a knot: a local linear support region that can move toward useful
mass or split when one region needs more resolution.

## Mathematical Grounding

Use sparse projections `u_m = p_m^T x`, initially coordinate-aligned plus a small
number of fixed sparse random projections. Each feature is a triangular hat
basis

```text
phi_m(x) = max(1 - |u_m - c_m| / s_m, 0),
```

with center knot `c_m` and support width `s_m`. Predictions are linear in the
expanded basis:

```text
y_hat_h(x) = b_h + sum_m w_{h,m} phi_m(x).
```

The readout can use the same LMS, IDBD, Autostep, or ObGD machinery already used
for Step 2 linear heads. The spline parameters receive local residual credit:

```text
delta_h = y_h - y_hat_h
g_c,m = sum_h delta_h w_{h,m} d phi_m / d c_m
c_m <- c_m + eta_c g_c,m
```

where `d phi_m / d c_m = sign(u_m - c_m) / s_m` inside support and `0` outside.
Each knot tracks EMA occupancy, squared residual contribution, and within-support
input variance. Every `split_interval`, split the highest-utility wide knot by
replacing the lowest-utility knot with a sibling at `c_m +/- s_m / 2`, halving
or shrinking both supports and copying half the outgoing weights. This is an
online adaptive approximation to free-knot linear splines: local approximation
error falls as knots concentrate where curvature or discontinuity is high.

## Why It Could Beat Previous Iteration And Fair MLP

The current fair MLP is expressive but diffuse: one residual event updates many
weights and can interfere with earlier regimes. A spline bank has compact
support, so credit assignment is local and old regions are less disturbed.
Compared with the previous Step 2 iteration, this gives an explicit feature
lifecycle without restricting the generator to pair products or random feature
search. Compared with a fair MLP, it may adapt faster after context changes
because new knots are allocated directly at persistent residual mass rather than
waiting for hidden units to reshape through nonconvex gradients.

The expected win condition is narrow but meaningful: nonstationary regression or
classification streams whose useful structure is low-dimensional after
projection and locally nonsmooth. It should not be claimed as a universal MLP
replacement.

## Minimal Implementation Sketch

Implement first as a standalone experiment learner, not shared core API. State:
`projection`, `center`, `width`, `utility_ema`, `occupancy_ema`,
`local_var_ema`, and linear readout state. Normalize observations with the
existing EMA normalizer. On every step: normalize `x`, compute active hats,
predict all heads, update readout, update knot centers by residual-gradient
credit, update EMAs, and use `jax.lax.cond` to split or refresh a knot every
fixed interval. Keep `M` active spline features fixed so parameter count can be
matched against `MultiHeadMLPLearner(hidden_sizes=(64,))` and a larger fair MLP
grid.

Initial streams: nonstationary one-dimensional and sparse-projection piecewise
linear targets, smooth sinusoidal targets, interaction stream as a non-primary
sanity check, and sklearn digits variants as external controls.

## Metrics / Success Bar

Primary metrics: final-window MSE, whole-stream prequential MSE, paired
seed-level wins, and adaptation lag after context switches. Secondary metrics:
held-out accuracy on digits, knot occupancy entropy, split frequency, and
fraction of active knots with nontrivial utility.

Promote only if adaptive splines beat the best fair MLP by at least `10%` mean
final-window MSE on two non-quadratic synthetic stream families, win at least
`10/12` paired seeds on the primary metric, and lose by no more than `3%` on
digits held-out accuracy or MSE. Include a parameter-matched MLP and a
larger-capacity MLP in the same run.

## Risks / Negative Controls

Risks: moving knots may chase noise, splitting may overfit recent residuals,
high-dimensional targets may not be low-dimensional after sparse projection,
and compact support may leave dead regions with no gradient.

Negative controls: frozen uniform knots with the same feature count; moving
knots with random residual signs; no-split moving knots; split-only frozen
centers; a purely linear stream where splines should not beat the linear
baseline materially; a smooth RFF-friendly sinusoid where frozen RFF should be
competitive; and the hashed-quadratic interaction stream to verify this is not
just another version of M04.

## Exact First Command

After adding the standalone experiment script, run:

```bash
source .venv/bin/activate && PYTHONPATH=src python "examples/The Alberta Plan/Step2/adaptive_spline_basis_experiment.py" --num-steps 3000 --seeds 12 --feature-dim 10 --n-tasks 2 --spline-features 96 --candidate-knots 192 --split-interval 50 --knot-lr 0.003 --output-dir outputs/step2_new_directions/d04_adaptive_spline_basis
```

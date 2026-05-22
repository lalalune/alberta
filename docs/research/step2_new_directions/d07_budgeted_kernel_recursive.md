# D07 Budgeted Kernel Recursive Learner With ALD Dictionary

## Core Hypothesis

A budgeted kernel recursive learner can beat the fair Step 2 MLP on small to
medium online nonlinear streams because it makes two commitments the MLP does
not: allocate new basis functions only when the current observation is genuinely
novel, and solve the resulting local regression problem with second-order
recursive updates. The prediction model is the kernel dictionary itself, not an
auxiliary correction on top of a neural predictor. If Step 2 failures are often
caused by slow feature creation and ill-conditioned first-order credit
assignment, an approximate Gaussian-process / KRLS learner should adapt in fewer
samples while keeping capacity bounded.

## Why Different

This is not centered targets, ECOC, random Fourier features, hashed quadratic
features, residual imprinting, a fast/slow ensemble, surprise reset, prototype
context, hidden-weight orthogonalization, or EMA prediction. RFF and hashed
quadratic methods use fixed feature maps. Residual imprinting stores residual
patches around a separate main learner. D07 instead makes the kernel dictionary
the main predictor, chooses dictionary elements by an approximate linear
dependency novelty test, and updates output weights with recursive least squares
or sparse GP posterior algebra. The central mechanism is online model-order
selection plus second-order coefficient adaptation.

## Mathematical Grounding

Let `z_t` be the online-normalized observation and use a bounded kernel such as

```text
k(z, z') = exp(-||z - z'||_2^2 / (2 sigma^2)).
```

The predictor for head `h` is

```text
y_hat_h(z_t) = k_D(z_t)^T alpha_h,
```

where `D = {c_1, ..., c_m}` is an online dictionary and
`k_D(z_t) = [k(c_1, z_t), ..., k(c_m, z_t)]`. Before adding `z_t`, compute its
approximate linear dependency score against the current dictionary:

```text
a_t = K_DD^-1 k_D(z_t)
nu_t = k(z_t, z_t) + lambda - k_D(z_t)^T a_t.
```

`nu_t` is the Schur-complement residual variance: in sparse GP language it is
the part of the prior variance not explained by the current inducing set; in
KRLS language it is the squared norm of the new kernel feature after projection
onto the dictionary span. Add `z_t` only when `nu_t > novelty_threshold` and the
budget allows it. If the dictionary is full, replace or merge the lowest-utility
center, using small leverage, low recent activation, low coefficient norm, and
age as the first simple score.

Given the current dictionary, update coefficients by RLS over kernel features:

```text
phi_t = k_D(z_t)
g_t = P_{t-1} phi_t / (rho + phi_t^T P_{t-1} phi_t)
e_t = y_t - alpha_{t-1}^T phi_t
alpha_t = alpha_{t-1} + g_t e_t
P_t = rho^-1 (P_{t-1} - g_t phi_t^T P_{t-1}).
```

For multiple heads, share `D`, `K_DD^-1`, and `P`, and update each active
`alpha_h` with its own error. This is the finite-dictionary form of online
kernel ridge regression with exponential forgetting. The ALD test controls
representation growth; `P_t` supplies the second-order curvature estimate that
first-order MLP heads and elementwise adaptive methods cannot represent.

## Why It Could Beat Previous Iteration And Fair MLP

The previous Step 2 direction established that a fair MLP is a strong baseline
when it has enough width and regularization, but it still learns new nonlinear
regions through diffuse gradient changes across many parameters. A Gaussian
kernel dictionary gives immediate local basis support around novel observations.
When the target changes by regime, class block, scale, or local discontinuity,
the ALD test should allocate capacity only where the current dictionary cannot
span the new region. The RLS update then re-solves the active kernel regression
problem in a few steps instead of relying on scalar or per-weight step-size
tuning.

This could beat residual imprint specifically because no separate slow learner
must first make a mistake and then be corrected. The kernel learner owns the
prediction, its novelty criterion is geometric rather than residual-only, and
its coefficient update uses covariance information. It should be strongest on
nonstationary low-data regimes where exact local memory and calibrated forgetting
matter more than extrapolating far outside observed support. A fair MLP remains
the right comparator because D07 should not win merely by adding more effective
parameters; it must win under a fixed dictionary budget and matched online data.

## Minimal Implementation Sketch

Implement first as a standalone experiment script, not core framework code:
`examples/The Alberta Plan/Step2/new_directions/d07_budgeted_kernel_recursive.py`.
Use existing streams, the same online normalization as fair MLP controls, paired
seeds, and fair MLP comparators.
Add a script-local `BudgetedKRLSState` with `centers`, `active_mask`,
`K_inv`, `P`, `alpha`, running activation, running coefficient magnitude, age,
and counters for adds/replacements.

At every time step: normalize `x_t`; compute `phi_t`; predict prequentially;
apply the ALD test; add a center if `nu_t` crosses threshold and budget is not
full; if full, replace the lowest-utility inactive or weak center and rebuild
`K_inv`/`P` with ridge jitter; then run the RLS coefficient update for active
heads. Start with Gaussian kernels over raw normalized inputs. Sweep
`budget in {32, 64, 128}`, `sigma` from median-distance warmup or
`{0.5, 1.0, 2.0}`, `rho in {0.97, 0.99, 0.995, 1.0}`,
`novelty_threshold in {1e-4, 1e-3, 1e-2}`, and ridge
`lambda in {1e-4, 1e-3, 1e-2}`. Keep all updates temporally uniform: novelty
statistics, replacement utilities, and RLS state update every step.

## Metrics / Success Bar

Primary metrics: paired final-window MSE or accuracy, whole-stream prequential
regret versus fair MLP, adaptation half-life after regime boundaries, and
held-out final-regime accuracy on sklearn digits variants. Kernel-specific
diagnostics: active dictionary size, add/replacement rate, ALD residual
distribution, mean predictive leverage `phi_t^T P phi_t`, coefficient norm,
effective memory under `rho`, non-finite count, and runtime multiplier.

Smoke success: over at least 5 paired seeds, beat the fair MLP final-window MSE
by `>= 5%` on two nonstationary synthetic regimes while losing no more than
`0.01` final-window accuracy on digits. Promotion success: 30 seeds, bootstrap
CI excluding zero on aggregate final-window improvement, paired win rate at
least `21/30` on the primary nonstationary metric, runtime below `4x` fair MLP
for `budget <= 128`, and no stationary IID regression or digits degradation
larger than `2%`.

## Risks / Negative Controls

Risks: Gaussian kernels may fail in high dimension without careful bandwidth;
dictionary replacement can destabilize `K_inv` and `P`; RLS can overfit noise
when `rho` is too low; local kernels extrapolate poorly outside observed support;
and rebuilding matrices at replacement time may dominate runtime. Numerical
jitter, Cholesky-based solves, and explicit finite checks are required from the
first prototype.

Negative controls: fixed random dictionary with the same RLS update, ALD
dictionary with first-order LMS coefficients, no-replacement dictionary capped
early, residual-triggered dictionary insertion without ALD, linear kernel KRLS,
shuffled-label stream, oversized fair MLP, and a stationary IID stream where the
method should match but not materially beat the MLP through leakage or target
memorization. A valid win requires the ALD plus RLS combination to beat both
fixed-dictionary RLS and ALD plus first-order coefficients.

## Exact First Command

After adding the standalone experiment script, run:

```bash
source .venv/bin/activate && PYTHONPATH=src python "examples/The Alberta Plan/Step2/new_directions/d07_budgeted_kernel_recursive.py" --steps 1500 --n-seeds 5 --final-window 300 --budgets 32 64 128 --sigmas 0.5 1.0 2.0 --rho 0.99 --novelty-thresholds 0.0001 0.001 0.01 --ridge 0.001 --output-dir outputs/step2_new_directions/d07_budgeted_kernel_recursive
```

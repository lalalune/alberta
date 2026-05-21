# D01 Online Newton Last-Layer Readout

## Core Hypothesis

The fair Step 2 MLP is often limited less by representational capacity than by slow,
ill-conditioned online output adaptation over its own hidden features. Replace only
the final linear readout update with an online second-order method: full recursive
least squares (RLS) when the last hidden width is small, or a diagonal
Gauss-Newton/online-Newton readout when memory must stay linear. If the hidden
features are already useful but correlated, the readout should track target and
context changes in tens of steps instead of waiting for scalar-step LMS/ObGD head
updates to crawl through the hidden-feature covariance.

## Why It Differs From The Prior Ten

This is not a target-code change, ECOC scheme, random or hashed lift, residual
dictionary, ensemble, reset rule, context/prototype augmentation, hidden-weight
regularizer, or parameter averaging method. The feature map remains the existing
MLP trunk; the intervention is exclusively the online optimizer for the last
linear layer. It tests a different claim: the current Step 2 weakness may be
readout conditioning over learned features, not missing feature construction.

## Mathematical Grounding

Let `h_t = f_theta(x_t)` be the current MLP trunk feature and
`phi_t = [h_t; 1]` include the bias feature. For each output head,
`y_hat_t = w_t^T phi_t`. With fixed `theta`, the best exponentially weighted
linear readout solves

```text
min_w 1/2 sum_{s <= t} rho^(t-s) (y_s - w^T phi_s)^2 + delta/2 ||w||_2^2.
```

RLS maintains the inverse covariance
`P_t = (sum rho^(t-s) phi_s phi_s^T + delta I)^(-1)` by Sherman-Morrison:

```text
k_t = P_{t-1} phi_t / (rho + phi_t^T P_{t-1} phi_t)
e_t = y_t - w_{t-1}^T phi_t
w_t = w_{t-1} + k_t e_t
P_t = rho^(-1) (P_{t-1} - k_t phi_t^T P_{t-1}).
```

For multi-head classification this is applied independently per active head, or
with one shared `P_t` over `phi_t` and a vector error update for all heads. The
diagonal variant keeps `d_i <- rho d_i + phi_i^2` and updates
`w_i <- w_i + eta e_t phi_i / (epsilon + d_i)`, which is the diagonal
Gauss-Newton preconditioner for squared error. Full RLS is exact for the
readout; diagonal GN is the cheap control.

## Why It Might Beat The Previous Iteration And Fair MLP

The previous Step 2 audits showed two relevant facts: fair MLP capacity removes
the old pair-product advantage, and context/output-adaptation probes can be
strongly limited by whether the readout can maintain the right slopes over
available features. A fair MLP head trained by LMS/ObGD sees gradients scaled by
`phi_t`; when hidden units are correlated or layer-normed into a narrow
subspace, one global or elementwise step-size cannot quickly solve the local
least-squares problem. RLS directly whitens that last-layer problem. It may
therefore preserve the fair MLP's external-data strength while reducing the lag
that makes UPGD or specialized context methods look better on synthetic
nonstationary streams.

## Minimal Implementation Sketch

Prototype only in a new experiment script, with no core API change. Reuse
`MultiHeadMLPLearner` initialization, trunk forward pass, normalizer, trunk
optimizer, and ObGD bounding. Add a sidecar readout state:

- `readout_w: Float[n_heads, hidden_dim + 1]`;
- either `P: Float[hidden_dim + 1, hidden_dim + 1]` shared across heads, or
  `diag: Float[hidden_dim + 1]`;
- hyperparameters `rho in {0.95, 0.98, 0.995, 1.0}`, `delta in {0.1, 1, 10}`,
  and optional `eta` for diagonal GN.

At each online step: compute pre-update `phi_t`, predict with `readout_w`, record
prequential errors, update the trunk by the existing semi-gradient path using the
same pre-update error and readout weights, then update the readout by RLS/GN for
active heads. Use the same initial trunk weights, stream order, hidden sizes,
normalization, and ObGD trunk settings as the fair MLP comparator. Include
`hidden_sizes=(64,)` and `(64, 64)`; full RLS is acceptable for 65- or
129-dimensional readouts.

## Metrics And Success Bar

Primary metrics: paired final-window MSE and accuracy, adaptation half-life
after label/context shifts, held-out final-phase accuracy on digits, and
prequential regret versus fair MLP. Diagnostics: `cond(P^-1)` or diagonal
spread, mean gain `phi_t^T P phi_t`, readout norm, trunk-gradient norm, NaN
rate, and runtime multiplier.

Smoke success: across at least 5 paired seeds, improve fair MLP final-window MSE
by `>= 5%` on one nonstationary synthetic/context stream and one external
digits-shift regime, with no digits accuracy loss larger than `0.005`. Promotion
success: 30 seeds, paired win rate at least `21/30` or bootstrap CI excluding
zero on the primary metric, while matching fair MLP on stationary shuffled
digits and not losing to the previous best non-portfolio Step 2 method on its
strongest synthetic stream.

## Risks And Negative Controls

RLS assumes a slowly changing feature basis; the MLP trunk moves, so stale
covariance can become wrong. Strong readout adaptation may also absorb all error
and starve the trunk of useful gradients. Full RLS is `O(H^2)` memory/time and
can become numerically brittle without ridge and forgetting.

Negative controls: frozen random trunk plus RLS, frozen trained trunk plus RLS,
fair MLP trunk with standard LMS/ObGD head, diagonal GN versus full RLS,
`rho=1.0` versus forgetting, shuffled-label stream, and a linear baseline on raw
inputs. A win only counts if RLS beats the same MLP features with a first-order
head, not merely an under-parameterized baseline.

## Exact First Command

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d01_online_newton_last_layer.py" --quick --seeds 0 1 2 3 4 --steps 1500 --output-dir outputs/step2_new_directions/d01
```

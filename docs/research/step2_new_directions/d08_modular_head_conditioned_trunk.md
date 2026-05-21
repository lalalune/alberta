# D08 Modular Head-Conditioned Trunk Modulation

## Core Hypothesis

The fair Step 2 MLP fails in multi-head continual settings partly because all
heads push through the same trunk coordinates with no learned way to route
conflicting gradients. Add a small per-head FiLM gate on the shared hidden
representation: each head sees an affine-modulated view of the same trunk
features, initialized as the identity. The trunk is still shared and updated at
every time step, but each head can learn which hidden units it should amplify,
suppress, or shift. This should reduce destructive sharing without replacing the
MLP with independent learners.

## Why Different

This is not a centered target, ECOC code, RFF lift, hashed quadratic feature map,
residual imprint dictionary, fast/slow ensemble, surprise reset, prototype
context append, weight orthogonalization rule, or EMA predictor. D08 changes the
conditional computation path inside the multi-head learner. The input, target
code, optimizer family, stream, and evaluation protocol stay fixed; only the way
head identity modulates shared hidden features changes.

## Mathematical Grounding

Let the shared trunk produce `h_t = f_theta(x_t) in R^H`. For head `k`, define
bounded FiLM parameters

```text
gamma_k = 1 + s_g tanh(a_k)
beta_k  = s_b tanh(c_k)
z_{t,k} = gamma_k * h_t + beta_k
y_hat_{t,k} = v_k^T z_{t,k} + b_k.
```

The identity initialization `a_k = c_k = 0` exactly reproduces the fair MLP at
step zero. With squared error and mask `m_{t,k}`, the trunk cotangent is

```text
dL_t / dh_t = sum_k m_{t,k} e_{t,k} (gamma_k * v_k),
e_{t,k} = y_hat_{t,k} - y_{t,k}.
```

In the current shared-head MLP, every head's trunk gradient is routed by `v_k`
alone. With FiLM, the effective head vector is `gamma_k * v_k`, so the learner
can reduce the interference term
`<gamma_i * v_i, gamma_j * v_j>` for heads whose errors anticorrelate, while
leaving aligned heads close to ordinary sharing. A small regularizer
`lambda_g ||gamma_k - 1||_2^2 + lambda_b ||beta_k||_2^2` keeps the mechanism
near the fair MLP unless the data pays for specialization. This is a constrained
mixture between full sharing and independent heads, not a new target geometry or
fixed feature expansion.

## Why It Could Beat The Previous Iteration And Fair MLP

The previous fair MLP comparison gives all heads the same hidden basis and folds
their cotangents into one VJP. That is efficient, but it assumes that every head
wants compatible trunk updates. In class-blocked digits, label drift, and
multi-cumulant streams, some heads are rare, stale, or temporarily misleading;
their gradients can erase features needed by other heads before those features
become useful again. D08 gives each head a low-cost learned routing vector, so a
rare or drifting head can preserve a subset of useful trunk units without
forcing a global reset, an ensemble split, or a hand-coded context signal.

It could beat the previous new-direction batch because it attacks a different
bottleneck: head-to-trunk credit assignment under shared representation
pressure. If fair MLP capacity is sufficient but the shared update geometry is
wrong, per-head modulation should improve adaptation while keeping the
stationary IID digits behavior close to baseline.

## Minimal Implementation Sketch

Implement first as an experiment-local wrapper, not a core API change:
`examples/The Alberta Plan/Step2/new_directions/d08_modular_head_conditioned_trunk.py`.
Copy the paired-seed structure from the existing Step 2 direction scripts.

- Start from the fair `MultiHeadMLPLearner` settings: same hidden sizes,
  sparse initialization, LayerNorm setting, normalizer, optimizer, ObGD
  bounding, stream order, and prequential scoring.
- Add script-local FiLM state with `a: Float[n_heads, hidden_dim]` and
  `c: Float[n_heads, hidden_dim]`, plus optimizer state for those arrays.
- Use `gamma = 1 + gate_scale * tanh(a)` with
  `gate_scale in {0.25, 0.5, 1.0}` and
  `beta = shift_scale * tanh(c)` with `shift_scale in {0.0, 0.1}`.
- Compute all head predictions from `z_{t,k}`. Apply NaN masks exactly as the
  multi-head learner does; inactive heads receive zero loss and zero gate
  gradient but the scan state still advances.
- Update trunk, heads, and FiLM parameters once per time step from the same
  prequential error. Use a smaller gate step-size grid
  `{0.1, 0.3, 1.0} * head_step_size` so gates do not immediately saturate.
- Compare against fair MLP, parameter-matched wider MLP, fixed random gates,
  identity gates, and a head-only FiLM control with `stop_gradient(h_t)` for the
  gate path.

## Metrics / Success Bar

Primary metrics: paired final-window MSE, final-window accuracy, held-out
final-regime accuracy where available, and adaptation half-life after drift or
class-block boundaries. Required diagnostics: effective head-vector cosine
matrix `<gamma_i * v_i, gamma_j * v_j>`, raw trunk-gradient cosine, gate
deviation `||gamma - 1||`, beta norm, per-head loss, update-bound scale, NaN
rate, and runtime multiplier.

Smoke success: over 5 paired seeds, D08 must improve aggregate final-window MSE
by at least `5%` versus the best fair MLP on two nonstationary multi-head
regimes, with no aggregate accuracy loss larger than `0.005`, no individual
regime accuracy loss larger than `0.02`, and a win over the parameter-matched
wider MLP. Promotion success: 30 paired seeds, bootstrap CI excluding zero on
the primary MSE improvement, paired win rate at least `21/30` in the strongest
regime, and a measured reduction in negative trunk-gradient cosine or effective
head-vector overlap.

## Risks / Negative Controls

Risks: the gate may act like extra readout capacity rather than solving trunk
interference; bounded gates can saturate and starve the trunk; beta shifts can
duplicate head biases; specialization can hurt regimes where heads genuinely
share features; and parameter-count gains can masquerade as a mechanism win.

Negative controls: `gate_scale=0` must reproduce fair MLP; fixed random gates
test whether any partition helps; shuffled head IDs test whether stable
head-conditioning matters; beta-only and gamma-only variants isolate the useful
part; frozen trunk plus FiLM tests whether the gain is only last-layer
adaptation; and IID shuffled digits should remain neutral. A positive result
only counts if D08 beats both identity-gate fair MLP and parameter-matched wider
MLP under identical seeds.

## Exact First Command

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d08_modular_head_conditioned_trunk.py" --steps 1500 --n-seeds 5 --final-window 300 --regimes iid label_drift class_blocked permuted_pixels --gate-scales 0.25 0.5 1.0 --shift-scales 0.0 0.1 --output-dir outputs/step2_new_directions/d08_modular_head_conditioned_trunk
```

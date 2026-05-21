# D06 Online Whitening Natural-Gradient Preconditioning

## Core Hypothesis

Fair Step 2 MLPs lose plasticity partly because hidden activations and
backpropagated update directions become highly correlated. A temporally-uniform
online preconditioner that whitens each layer's incoming activations and
loss-gradient directions should make a simple MLP update closer to a local
natural gradient, reducing interference without changing targets, adding
features, resetting parameters, ensembling learners, or orthogonalizing weights.

## Why Different

This is not M09 hidden-weight orthogonalization. M09 periodically replaced the
first-layer weight matrix geometry. D06 leaves all weights as learned and only
changes the metric used to apply gradients. It is also distinct from centered
targets, ECOC, RFF, hashed quadratic features, residual imprinting, fast/slow
ensembles, surprise resets, prototype contexts, and EMA prediction because it
does not alter labels, inputs, model capacity, model count, update schedule, or
parameter averaging. The mechanism is optimizer geometry: online estimates of
activation and gradient covariance are used to precondition every step.

## Mathematical Grounding

For layer `l`, let `a_l` be the incoming activation vector and let `u_l` be the
loss-gradient cotangent at that layer's preactivation. The instantaneous weight
gradient has rank-one form `G_l = u_l a_l^T` for a dense layer, including layers
followed by LayerNorm/LeakyReLU when `u_l` is taken before the affine map. Keep
EMA covariance estimates:

```text
A_l <- (1 - eta) A_l + eta a_l a_l^T
B_l <- (1 - eta) B_l + eta u_l u_l^T
```

with damping `lambda I`. A Kronecker-factored empirical Fisher approximation is
`F_l ~= A_l otimes B_l`, so a natural-gradient-like update for a weight trace
`Z_l` is:

```text
D_l = (B_l + lambda I)^(-p) Z_l (A_l + lambda I)^(-p)
```

where `p=1` is the K-FAC-style inverse and `p=0.5` is the safer whitening
variant for the first smoke test. Bias traces use only the left factor. This
directly attacks covariance-induced conditioning while preserving the ordinary
MLP function class.

## Why It Could Beat Previous Iteration And Fair MLP

The fair MLP and ObGD bounding control update magnitude, but not update
direction conditioning. IDBD/Autostep adapt elementwise step-sizes, which cannot
remove cross-feature covariance. M09 showed that decorrelating weights can raise
rank but hurt performance, suggesting the representation geometry itself should
not be forcibly rewritten. D06 keeps useful learned geometry and instead makes
the update invariant to correlated activations. It should help most on
class-blocked and permuted-pixel digits, where recent features are correlated
and stale gradients interfere, while remaining neutral on IID digits if damping
and ObGD bounding are doing their job.

## Minimal Implementation Sketch

Implement as an experiment-only script, e.g.
`examples/The Alberta Plan/Step2/new_directions/d06_online_whitening_natural_gradient.py`,
copying the paired-stream structure from the moonshot scripts but not changing
core framework code.

- Use `MultiHeadMLPLearner` with the same fair baseline settings as recent
  digits moonshots: `hidden_sizes=(64,)`, `step_size=0.03`,
  `ObGDBounding(kappa=2.0)`, `sparsity=0.5`, `use_layer_norm=True`, and shared
  initialization/stream per seed.
- Add a script-local `WhiteningState` with per-trunk-layer `A_l`, `B_l`,
  inverse factors, decay `eta`, damping `lambda`, and `power p`.
- In a script-local learner wrapper, copy the current multi-head update path.
  Recover or record layer activations during the forward pass. For trunk layers,
  use the VJP-produced loss-weighted gradients; recover `u_l` from
  `G_l ~= u_l a_l^T` using `u_l = G_l a_l / (a_l^T a_l + eps)` when needed.
- Update all covariance states every time step. Compute inverse powers with
  `jnp.linalg.eigh`, eigenvalue floor `lambda`, and clipping on the resulting
  preconditioned step norm.
- Precondition traces before optimizer/bounder calls: trunk uses
  `D_l = B_l^-p Z_l A_l^-p`; heads use the hidden activation covariance on the
  right side and scalar left factor. Keep ObGD bounding after preconditioning.
- Compare against fair MLP, diagonal-only preconditioning, and identity
  preconditioner under identical seeds.

## Metrics / Success Bar

Primary metrics: paired final-window MSE and final-window accuracy on sklearn
digits `iid`, `label_drift`, `class_blocked`, and `permuted_pixels`, plus
held-out final-regime accuracy where the existing scripts support it. Secondary
metrics: online mean MSE, adaptation half-life after drift boundaries, update
bound scale, covariance condition numbers, preconditioned step norm, and NaN /
non-finite counts.

Promote only if full whitening with `p=0.5` beats fair MLP final-window MSE in
at least 75% of paired seed-regime cells, improves aggregate final-window MSE by
at least 3%, has no aggregate final-window accuracy drop larger than `0.005`,
has no individual nonstationary cell accuracy drop larger than `0.02`, and beats
the diagonal-only preconditioner on at least two nonstationary regimes.

## Risks / Negative Controls

Risks: small-batch covariance estimates can be noisy; inverse square roots can
amplify rare directions; LayerNorm may make activation covariance less
informative; eigendecompositions add meaningful per-step compute; preconditioning
may fight IDBD/Autostep meta-updates if combined too early. Start with LMS plus
ObGD only.

Negative controls: `p=0` identity preconditioning, diagonal-only `A_l`/`B_l`,
feature-only whitening, gradient-only whitening, frozen covariance after warmup,
and IID digits where the method should not create a gain by simply destabilizing
the baseline. Also log M09-style first-layer rank/cosine diagnostics to confirm
any win is not secretly caused by weight orthogonalization.

## Exact First Command

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d06_online_whitening_natural_gradient.py" --steps 1200 --n-seeds 3 --final-window 300 --regimes iid label_drift class_blocked permuted_pixels --precondition-power 0.5 --cov-decay 0.01 --damping 0.03 --output-dir outputs/step2_new_directions/d06_online_whitening_natural_gradient
```

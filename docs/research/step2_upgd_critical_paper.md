# Target-Structure UPGD for Step 2: Critical Case for Promotion

Date: 2026-05-07.

## Abstract

The current Step 2 evidence supports a single no-portfolio learner built around
utility-based perturbed gradient descent (UPGD), online feature utility,
bounded updates, and low-utility perturbation. The strongest current default in
the repo is target-structure UPGD, exposed as
`UPGDLearner.step2_default(n_heads)`. It is now the resource-efficient branch:
hidden size 32, `ObGDBounding(kappa=0.5)`, low-noise Rademacher perturbation
every 16 steps, target-structure loss, and no default adaptive-kappa,
meta-plasticity, unit-recycling, or gradient-history traces.

Target-structure UPGD bridges dense vector regression and sparse one-hot
classification by using sum-style loss only for non-negative simplex targets
with mass one, and mean-style loss otherwise. This supersedes the interim
target-density UPGD interpretation, which divided by the number of nonzero
supervised targets and therefore confused exact-zero dense regression and
sparse multilabel supervision with one-hot classification.

The key claim is deliberately conditional, not universal in the no-free-lunch
sense. Target-structure UPGD is canonical for the Step 2 matrix if its target
normalizer preserves the update scale of dense regression and one-hot
classification, while UPGD's utility perturbation and optional unit recycling
preserve an online mechanism for reallocating low-utility nonlinear features.
The exact-zero dense and sparse multilabel stressors have now been tested and
favor the target-structure rule. The 30-seed confirmation clears all three
synthetic out-of-class streams, and the adaptive structure rerun improves the
five-regime sklearn-digits pressure matrix. The external 10-seed matrix also
shows that the strongest current learner is not universally superior:
same-run MLP baselines still win scalar tabular regression and delayed/history
temporal regression. The strict digit readout conflict has since been closed
by a heavier 64-64 two-timescale readout branch that beats the best same-run
fair MLP on all five 30-seed digit final-window MSE rows. A direct synthetic
rerun keeps this branch out of the global default role because it loses the
compositional regression stream.

## Critical Thesis

Target-density UPGD is a strong empirical bridge, but it has the wrong
invariant. It treats "few nonzero target entries" as the same thing as
"classification-style sparse supervision." That is false for at least two
important cases:

- Dense regression with exact-zero coordinates.
- Sparse multilabel targets where multiple positive labels can be correct.

The cleaner invariant is target structure:

- Non-negative simplex targets with mass one are classification-like, so they
  should keep sum-style per-head pressure.
- All other active target vectors are regression/multilabel-like, so they
  should use mean-style scaling over active heads.

This is not a dataset router. It is a local property of the target vector
observed at the current time step, so it remains causal and temporally uniform.

## Conditional Theorem

Let `A_t` be the active target coordinates at time `t`, and let
`y_t[A_t]` be the active target vector. Define:

- `simplex(y_t)`: all active entries are non-negative, the active target mass
  is positive, and the active target mass equals one up to numerical tolerance.
- `d_t = 1` if `simplex(y_t)`, otherwise `d_t = |A_t|`.
- `L_t(theta) = 0.5 * sum_{i in A_t} (f_i(x_t; theta) - y_{t,i})^2 / d_t`.

Then target-structure UPGD has the following scale invariance:

1. For one-hot classification encoded as a simplex vector, `d_t = 1`; the
   update equals sum-loss multihead squared error and avoids diluting the
   positive class signal by the number of inactive classes.
2. For dense regression, including coordinates that are exactly zero,
   `d_t = |A_t|`; the update equals mean-loss vector regression and keeps the
   gradient scale independent of output dimension.
3. For sparse multilabel targets with mass not equal to one, `d_t = |A_t|`;
   the update avoids the target-density failure mode that can over-scale rows
   only because many supervised labels are zero.
4. If per-step gradients are bounded by ObGD and meta-plasticity multipliers
   are clipped, the per-step parameter displacement is bounded by the chosen
   `kappa` envelope and the configured multiplier interval.
5. If `perturbation_sigma > 0`, then any hidden weight with normalized utility
   below one receives nonzero variance perturbation on perturbation steps.
   Low-utility hidden features therefore retain an exploration channel while
   high-utility features are perturbed less.

This theorem explains update-scale consistency and sustained low-utility
feature exploration. It does not prove universal recursive representation
learning, global convergence, or dominance over all MLPs on all distributions.
Such a universal claim would contradict standard no-free-lunch limitations.

## Theorem Assumptions and Limits

The conditional theorem above assumes bounded observations, finite active
target sets, finite network weights before each update, clipped ObGD/meta
multipliers, finite loss-normalization denominators, and independent
zero-mean perturbation noise with finite variance. The low-utility exploration
statement is an expected-energy statement about the perturbation scale, not a
guarantee that a sampled perturbation improves utility or discovers the right
feature.

The rank and utility diagnostics are falsification tools, not proof
obligations. Stable/effective rank can show hidden-feature collapse or
retention, utility distributions can show whether exploration is targeted, and
replacement ages/counts can show churn. None of these metrics alone proves
recursive feature construction, optimal allocation of hidden units, or
dominance over a fair fixed-width MLP.

## Known Work

UPGD and continual backprop are known mechanisms from the Alberta continual
learning line. Dohare et al. show that deep networks can lose plasticity under
continual streams and that L2 regularization plus weight perturbation, and then
continual backprop, can maintain plasticity by reinitializing or perturbing
less useful units. The repo's UPGD implementation inherits this core idea:
estimate utility online and inject randomness where utility is low.

Recent plasticity work has sharpened the diagnosis. Plasticity loss is linked
to dormant units, rank/effective-rank collapse, Neural Tangent Kernel rank
decrease, and excessive churn. These papers support the use of utility,
feature-rank diagnostics, bounded updates, and careful stability/plasticity
controls as necessary ablations, not optional presentation details.

What appears novel in this repo is not UPGD itself. The novel claim is the
Step 2-specific combination:

- A single vector-output UPGD learner for dense regression and sparse
  classification streams.
- Target-structure loss normalization as a causal, per-example bridge between
  regression and one-hot supervision.
- Resource-efficient low-utility perturbation with intervaled bounded
  Rademacher mutation.
- Explicit feature-utility and hidden-resource management inside the same
  continually updated nonlinear learner, with heavier unit-recycling and
  meta-plasticity mechanisms kept as ablations rather than defaults.

## Current Empirical Gate

The current paper/deployment gate is executable:

```bash
python benchmarks/step2_upgd_evidence_gate.py
```

It passes when:

- `UPGDLearner.step2_default` still serializes to the promoted
  target-structure, width-32, interval-16 Rademacher branch.
- The default state does not allocate disabled unit-recycling or
  previous-gradient buffers.
- The 30-seed synthetic out-of-class benchmark has positive paired diffs and
  all-seed wins on polynomial, frequency, and compositional streams.
- The 30-seed digits matrix clears final-window MSE, held-out MSE, held-out
  accuracy, and class-blocked caveat thresholds.
- The throughput artifact shows width-32 UPGD faster than MLP64 on one-hot and
  dense targets with no more trainable parameters or float state.

Current headline numbers:

| Evidence | Result |
|---|---|
| Synthetic polynomial | `+0.5634 +/- 0.0320`, `30/30` wins vs best MLP |
| Synthetic frequency | `+0.6107 +/- 0.0410`, `30/30` wins vs best MLP |
| Synthetic compositional | `+0.0781 +/- 0.0036`, `30/30` wins vs best MLP |
| Digits final-window MSE | `+0.0078 +/- 0.0004`, `147/150` wins vs MLP64 |
| Digits test accuracy | `+0.0269 +/- 0.0018`, `135/150` wins vs MLP64 |
| One-hot throughput | width-32 UPGD `1.45x` faster than MLP64 in the recorded benchmark |
| Dense throughput | width-32 UPGD `1.57x` faster than MLP64 in the recorded benchmark |

Current external falsifiers and rescues:

| Evidence | Result |
|---|---|
| External breadth vs MLP64 | UPGD wins final-window MSE on `7/10` domains |
| External breadth vs MLP64_64 | UPGD remains strongest on image-like, tabular classification, dense-zero, and multilabel rows |
| Diabetes regression | `upgd_reg_passthrough_deep` beats both MLP64 and `mlp_deep` by final-window MSE |
| Delayed-history temporal regression | passthrough scalar UPGD beats MLP64 by final-window and test MSE |
| Strict digit readout | two-timescale 64-64 UPGD beats the best same-run fair MLP on IID, class-blocked, label-drift, mask-noise, and permuted final-window MSE at 30 seeds |
| Class-blocked digits | strict two-timescale UPGD beats `MLP(64,64)` final-window MSE by `+0.000189` with `26/30` wins and retained accuracy by `+0.424985` |
| Label-drift digits | strict two-timescale UPGD beats `MLP(64,64)` final-window MSE by `+0.005352` with `30/30` wins |
| Strict readout synthetic check | positive on polynomial and frequency, negative on compositional regression, so not a global default |

## Holes To Close

The current evidence has four main holes.

1. The current theory explains scale and exploration, not universal feature
   construction. A stronger theorem would need assumptions about stream
   recurrence, feature budgets, approximation class, and nonstationarity.
2. External online tasks now exist in the repo. Scalar tabular regression and
   delayed/history temporal regression are rescued by passthrough scalar UPGD
   variants, but those variants damage multiclass/vector rows. The open
   question is therefore a causal target-structure readout rule, not whether a
   single hand-picked passthrough branch should replace the default.
3. The strict digit readout uses hand-set fast-trunk, fast-head, and slow-head
   budget multipliers. It is still one learner state, but a stronger canonical
   version should learn or derive those controls causally, especially because
   the digit-tuned row loses synthetic compositional regression.
4. The current falsification and sieve probes are negative for overbroad
   theorem language: delayed utility, rare targets, dense-zero dilution,
   unobserved drift, outside-composition, and finite-resource rows can all
   defeat the learner unless their assumptions are stated explicitly.

## Required Ablations

Promotion ablations:

- `mean`, `sum`, `target_density`, `target_structure`.
- `sigma=0` vs `sigma=1e-4` to isolate perturbation from normalization.
- ObGD off, fixed `kappa=0.5`, adaptive loss-ratio kappa, learned
  gradient-alignment kappa.
- Meta-plasticity off, head-only learned plasticity, trunk+head learned
  plasticity.
- Repetition head plasticity off, `0.25`, `0.75`, learned multiplier.
- Unit recycling off, loss-gated recycling, loss-pressure budget recycling.
- `readout_mode="linear_mse"` vs `softmax_ce` for simplex targets.
- `readout_mode="two_timescale_simplex"` with shared vs separately bounded
  fast head, fast-trunk CE pressure, and slow simplex MSE suppression.
- `readout_input_mode="hidden"` vs `hidden_plus_input`.

Diagnostics required for every run:

- Final-window MSE and online mean MSE.
- Held-out accuracy/MSE where a retained test set exists.
- Utility distribution by layer.
- Fraction of perturbation energy assigned to low-utility weights.
- Effective rank or stable rank of hidden activations.
- Learned kappa and learned group plasticity multipliers over time.
- Unit replacement counts and replacement ages when recycling is enabled.

## Domain Matrix

The full superiority claim should be constrained to online, one-pass, no-replay
MLP tasks. The next matrix should include:

- Existing synthetic polynomial, frequency, and compositional streams.
- Dense exact-zero regression and sparse multilabel stressors.
- Sklearn digits IID, permuted, class-blocked, label-drift, and mask-noise.
- OpenML tabular classification/regression streams already supported by the
  repo's external benchmark scripts.
- A small image-like stream with nonstationary pixel permutations.
- A temporal stream with delayed or history-dependent targets.

The fair MLP baseline must match hidden sizes, sparsity, layer norm, step size,
and ObGD bounding, with width sweeps reported instead of a single weak MLP.

## Canonical Promotion Bar

The target-structure rule is now the promoted Step 2 normalization default
because it satisfies the local promotion bar:

- It preserves the positive synthetic and one-hot digit evidence of the
  target-density branch where the target structures coincide.
- It fixes dense exact-zero and sparse multilabel stressors where
  target-density has the wrong invariant.
- Keeps class-blocked retained-test behavior inside the current conservative
  compromise, and records the 64-64 two-timescale readout closeout that beats
  the best same-run fair MLP on all five digit final-window MSE rows.
- Separates the normalization claim from the stronger learner claim: the
  simple fixed-kappa row is not sufficient against the stricter five-regime
  digits matrix with the wider fair MLP baseline.
- Treats scalar passthrough as a successful targeted rescue, but not as a
  canonical default until it can be selected by a target-structure rule without
  giving back the vector/multilabel/digit wins.
- Treats fixed softmax readout as a successful class-blocked rescue but not a
  universal digit default. The promoted strict digit readout is the bounded
  two-timescale branch that also preserves the linear-MSE label-drift win.
- Keeps the strict digit readout separate from the global `step2_default`
  factory because the strict row does not preserve the compositional synthetic
  win.

The repo should still avoid theorem language. The correct wording is that
target-structure UPGD closes the current normalization ambiguity and preserves
the strongest one-hot digit evidence by equivalence, not that it proves
universal representation learning or universal superiority over fair MLPs.

## References

- Dohare et al., "Continual Backprop: Stochastic Gradient Descent with
  Persistent Randomness", arXiv:2108.06325.
- Dohare et al., "Utility-based Perturbed Gradient Descent: An Optimizer for
  Continual Learning", arXiv:2302.03281.
- Dohare et al., "Maintaining Plasticity in Deep Continual Learning",
  arXiv:2306.13812 and Nature 2024.
- Recent plasticity-loss work connecting plasticity to churn, dormant units,
  rank collapse, and effective-rank regularization should be cited in any paper
  draft and used to motivate feature-rank diagnostics.

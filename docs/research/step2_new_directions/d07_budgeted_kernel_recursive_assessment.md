# D07 Budgeted Kernel Recursive Learner Assessment

## Question

The skeptical bar was not "can a router over MLPs close a benchmark?"  The bar
was whether a single learner with a more mathematical representation and
bounded computation can objectively beat the fair MLP baseline across Step 2.
D07 tests that directly with an experiment-only learner:

- a bounded ALD dictionary;
- Schur-complement novelty for feature allocation;
- RLS coefficient updates;
- optional diffusion/Green, polynomial, and arc-cosine/NNGP kernels;
- optional temporal center throttling, which reserves capacity for later
  regimes instead of spending the dictionary at the beginning of the stream.

This is not a router over MLP predictions.  The kernel dictionary owns the
prediction.

## Implementation

Script:

```bash
examples/The Alberta Plan/Step2/new_directions/d07_budgeted_kernel_recursive.py
```

Main additions:

- `green` and `gaussian` heat kernels with dimension-normalized distances;
- normalized and raw inhomogeneous polynomial kernels;
- `algebraic_green`, a direct-sum polynomial plus diffusion RKHS;
- `arccosine`, a finite-depth ReLU NNGP / arc-cosine kernel;
- `algebraic_arccosine`, a direct-sum polynomial plus NNGP kernel;
- ALD center throttling via `--center-add-intervals`;
- optional residual hybrid experiments, which were tested and rejected.

Verification:

```bash
ruff check .
MYPYPATH="src:examples/The Alberta Plan/Step2" mypy \
  "examples/The Alberta Plan/Step2/new_directions/d07_budgeted_kernel_recursive.py"
pytest tests/ -v
```

Results: ruff passed, targeted mypy passed, and the full test suite passed
`1046 passed, 36 warnings`.

## Key Results

### Pure Heat Kernel Failed

Output:

```text
outputs/step2_new_directions/d07_sweep_controlled_2seed/results.json
```

The first pure diffusion/Green kernel lost all controlled tasks.  Diagnosis was
clear: with a low ALD threshold the dictionary filled and then replaced nearly
every step, repeatedly resetting RLS covariance.  Example:

- `controlled_nonlinear`, `green_rls_b64_s0p5_r0p99_n0p0001`:
  `adds=64`, `replacements=536` over 600 steps, final-window MSE `0.2979`
  vs best MLP `0.0983`.

Conclusion: local memory alone is not the universal learner.

### Algebraic-Green RKHS Has Real Signal

Output:

```text
outputs/step2_new_directions/d07_algebraic_all_3seed_1200/results.json
```

Fixed family:

```bash
--kernel algebraic_green --algebraic-weight 0.75 --polynomial-degree 3 \
--budgets 128 --sigmas 0.5 1.0 --rhos 1.0 \
--novelty-thresholds 0.001 0.05 --no-replace-when-full
```

Best-kernel vs best-MLP, 3 seeds, 1200 steps:

- `controlled_nonlinear`: `+0.0273` final-window MSE diff, wins `3/0/0`.
- `controlled_interaction`: `+0.2285`, wins `3/0/0`.
- `controlled_triple`: `+0.4043`, wins `3/0/0`.
- `controlled_rare`: `+0.0293`, wins `3/0/0`.
- `controlled_polynomial`: `+0.5204`, wins `3/0/0`.
- `synthetic_polynomial`: `+0.0165`, wins `2/1/0`.
- `digits_iid`: test accuracy `+0.0303`, wins `3/0/0`.
- `digits_class_blocked`: test accuracy `+0.6797`, wins `3/0/0`.

Losses:

- `controlled_frequency`: `-0.1479`, loses `0/3/0`.
- `synthetic_compositional`: `-0.7858`, loses `0/3/0`.
- `synthetic_frequency`: `-0.1921`, mixed but mean negative.
- `digits_label_drift`, `digits_mask_noise`, `digits_permuted_pixels` lost
  before center throttling.

Conclusion: algebraic RKHS plus RLS is objectively better than MLP on many
recursive/algebraic and retention regimes, but not universal.

### Temporal Center Throttling Fixed Stateful Digits

Outputs:

```text
outputs/step2_new_directions/d07_label_drift_interval_3seed/results.json
outputs/step2_new_directions/d07_digits_permute_mask_interval_3seed/results.json
```

The successful resource-management change was simple but important: reserve
dictionary capacity by accepting centers only every `k` steps.  This prevents
the model from spending all 128 centers in the first phase.

Results:

- `digits_label_drift`:
  - final-window MSE diff `+0.0060`, wins `3/0/0`;
  - held-out test MSE diff `+0.0106`, wins `3/0/0`;
  - held-out test accuracy diff `+0.0439`, wins `3/0/0`.
- `digits_mask_noise`:
  - final-window MSE diff `+0.0048`, wins `3/0/0`;
  - held-out test MSE diff `+0.0065`, wins `3/0/0`;
  - held-out test accuracy diff `+0.0476`, wins `3/0/0`.
- `digits_permuted_pixels`:
  - final-window MSE diff `+0.0133`, wins `3/0/0`;
  - held-out test MSE diff `+0.0100`, wins `3/0/0`;
  - held-out test accuracy diff `+0.0402`, wins `3/0/0`.

Conclusion: a single RKHS learner with temporally managed ALD allocation beats
best fair MLP on the hard stateful digits regimes that previously blocked Step
2-style claims.

### Raw Polynomial RKHS Fixed Controlled Frequency

Output:

```text
outputs/step2_new_directions/d07_rawpoly_remaining_3seed/results.json
```

Best raw degree-3 polynomial kernel:

- `controlled_frequency`: final-window MSE `0.0030` vs best MLP `0.1569`;
  paired diff `+0.1538`, wins `3/0/0`.

This makes sense: the controlled frequency target is a cubic identity in the
provided sine coordinate.  The normalized polynomial kernel hid that exact
finite-degree structure; the raw polynomial kernel restored it.

But the same raw polynomial RKHS still lost:

- `synthetic_compositional`: best raw-poly MSE around `0.58` to `0.68` vs MLP
  `0.2758`.

Conclusion: raw tensor algebra solves the exact cubic frequency task, but it is
not enough for tanh-compositional structure.

### NNGP / Arc-Cosine Kernel Helped But Did Not Beat MLP

Outputs:

```text
outputs/step2_new_directions/d07_arccosine_comp_depth1_3seed/results.json
outputs/step2_new_directions/d07_arccosine_comp_b256_3seed/results.json
```

The arc-cosine kernel was aimed at the one stubborn target: the two-hidden-layer
tanh compositional stream.

Results:

- budget 128, depth 1: best final-window MSE `0.4272`;
- budget 256, depth 1: best final-window MSE `0.3725`;
- best MLP: `0.2758`;
- best-kernel vs best-MLP diff at budget 256: `-0.0985`, loses `0/3/0`.

Conclusion: infinite-width neural kernels are closer than algebraic/diffusion
kernels on compositional tanh, but still not enough at the tested budgets.

### Residual MLP + Kernel Hybrid Was Rejected

Outputs:

```text
outputs/step2_new_directions/d07_hybrid_probe_3seed/results.json
outputs/step2_new_directions/d07_hybrid_rawpoly_freq_comp_3seed/results.json
```

The hybrid was a single additive predictor, not a router: `prediction =
MLP + KRLS residual`.  It was tested because it could have combined MLP's smooth
compositional bias with RKHS retention and algebraic memory.

It did not clear the bar:

- on `synthetic_compositional`, `hybrid_mlp_h128 + algebraic_green` got
  `0.3050`, worse than MLP `0.2758`;
- on `controlled_frequency`, `hybrid_mlp_h64_64 + raw polynomial` got `0.1784`,
  worse than MLP `0.1569` and much worse than pure raw polynomial `0.0030`;
- on digits, hybrids often underperformed the pure RKHS learner.

Conclusion: residualizing on top of MLP is not the answer, and it also supports
the original skepticism: adding MLP does not automatically make a universal
learner.

## Current Bottom Line

D07 is a strong new direction, not a completed Step 2 universal learner.

Solved by D07:

- A single non-MLP RKHS learner can beat fair MLP on controlled nonlinear,
  interaction, triple, rare, polynomial, raw controlled frequency, synthetic
  polynomial, digits IID, class-blocked retention, label drift, mask noise, and
  permuted pixels.
- The decisive mechanism is not "more MLP."  It is RKHS structure plus RLS plus
  temporally distributed ALD resource allocation.
- Runtime at budget 128 is usually in the same order as the fair MLP for these
  1200-step probes; residual hybrids are slower and worse.

Still missing:

- No single fixed D07 configuration beats best fair MLP on every benchmark.
- The compositional tanh stream remains the clearest blocker: best D07 result
  tested was `0.3725` vs MLP `0.2758`.
- There is a real configuration conflict:
  - raw polynomial, immediate allocation solves controlled frequency;
  - normalized/distributed algebraic-Green solves stateful digits;
  - arc-cosine helps compositional but does not yet beat MLP.

## Next Concrete Direction

The next serious candidate should be a multi-bank single learner, not a router:

- bank 1: raw degree-3 polynomial RKHS with immediate allocation for exact
  algebraic and frequency structure;
- bank 2: normalized algebraic-Green RKHS with center throttling for retention
  and stateful external streams;
- bank 3: arc-cosine or learned operator/tanh bank for compositional smooth
  structure;
- a learned resource manager allocates center budget and RLS regularization
  across banks using loss improvement minus compute cost.

The prediction should be one additive model with a single loss, not a
prediction router.  Promotion requires one canonical multi-bank configuration
to beat best fair MLP across all Step 2 datasets without per-dataset selection.

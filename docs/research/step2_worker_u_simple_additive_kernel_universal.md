# Worker U: Simple Additive/Kernel Continuation

## Scope

Worker U inspected the D07/D08/D18 Step 2 continuation artifacts and ran a
focused confirmation for the strongest simple additive/kernel candidate under
the constraint: no portfolio, no prediction router, no MLP expert, and no
hand-engineered feature-search story as the answer.

The best empirical candidate is the D18 canonical-equivalent configuration:

```text
d18_gain_safecore_poly_unified_0p01
```

This is equivalent to the documented `step2_canonical` setting when run with
`--gain-step-size 0.2 --gain-l2 0.05 --component-clip 1.0`.

## Inspection Result

D08 is the cleanest implementation of the permitted form. It concatenates active
features from multiple banks and updates one shared output coefficient matrix
from one prequential prediction error. The strongest D08 prior result was the
full sequential multibank family, especially `canonical_green_first_full_seq`
and `canonical_full_seq`. Those runs showed that a single additive output can
beat fair MLP on compositional/frequency and some stateful digit regimes, but
the D08 outputs remained uneven on digit MSE/test-MSE and did not cover the full
weak matrix including `controlled_nonlinear`, `controlled_rare`, and
`synthetic_polynomial`.

D18 is the best empirical continuation. It adds:

- D10 resource-managed RKHS core: raw polynomial, algebraic-Green, and
  arccosine banks.
- D15 groupwise tanh/Fourier basis block.
- strict degree-3 polynomial RLS residual block.
- D14 unified residual basis.
- learned per-block gains anchored by L2 and clipped residual-channel outputs.

This remains additive at prediction time and has no MLP expert or hard output
router. However, it is not a clean "one shared alpha over all features" learner:
it learns block gains and trains component blocks against residual targets. This
is the central simplicity concern.

## Focused Confirmation Command

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" \
  --datasets controlled_nonlinear,controlled_rare,synthetic_compositional,synthetic_polynomial,synthetic_frequency,digits_permuted_pixels,digits_mask_noise,digits_label_drift \
  --steps 1200 \
  --n-seeds 3 \
  --final-window 300 \
  --configs gain_safecore_poly_unified_0p01 \
  --gain-step-size 0.2 \
  --gain-l2 0.05 \
  --component-clip 1.0 \
  --output-dir outputs/step2_worker_u_d18_canonical_equiv_weak_matrix_3seed \
  --note-path outputs/step2_worker_u_d18_canonical_equiv_weak_matrix_3seed/RUN_SUMMARY.md
```

Artifacts:

- `outputs/step2_worker_u_d18_canonical_equiv_weak_matrix_3seed/results.json`
- `outputs/step2_worker_u_d18_canonical_equiv_weak_matrix_3seed/SUMMARY.md`
- `outputs/step2_worker_u_d18_canonical_equiv_weak_matrix_3seed/RUN_SUMMARY.md`

Wall clock: `60.195302` seconds.

## Focused Confirmation Results

MSE diff is paired best-D18-vs-best-MLP, positive favors D18. Digit test
accuracy diff is also paired, positive favors D18.

| Dataset | D18 final MSE | Best MLP final MSE | MSE diff | MSE W/L/T | D18 test acc | Best MLP test acc | Test acc diff | Test acc W/L/T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `controlled_nonlinear` | 0.017445 | 0.057959 | +0.040513 | 3/0/0 | | | | |
| `controlled_rare` | 0.018128 | 0.073166 | +0.055038 | 3/0/0 | | | | |
| `synthetic_compositional` | 0.240080 | 0.275842 | +0.031623 | 3/0/0 | | | | |
| `synthetic_polynomial` | 0.798105 | 0.947464 | +0.149360 | 3/0/0 | | | | |
| `synthetic_frequency` | 0.882529 | 1.149273 | +0.265933 | 3/0/0 | | | | |
| `digits_permuted_pixels` | 0.037868 | 0.049263 | +0.011296 | 3/0/0 | 0.925170 | 0.881262 | +0.043908 | 3/0/0 |
| `digits_mask_noise` | 0.044755 | 0.047837 | +0.003082 | 3/0/0 | 0.827458 | 0.808905 | +0.007421 | 2/1/0 |
| `digits_label_drift` | 0.034849 | 0.038250 | +0.003288 | 3/0/0 | 0.953618 | 0.904143 | +0.046382 | 3/0/0 |

Focused result: green on all 8 requested datasets by final-window MSE, with
positive held-out digit accuracy on all three digit regimes.

## All-Suite Follow-Up Command

Because the focused matrix was green, I ran the same fixed candidate on the
script's `all` alias.

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 3 \
  --final-window 300 \
  --configs gain_safecore_poly_unified_0p01 \
  --gain-step-size 0.2 \
  --gain-l2 0.05 \
  --component-clip 1.0 \
  --output-dir outputs/step2_worker_u_d18_canonical_equiv_all_3seed \
  --note-path outputs/step2_worker_u_d18_canonical_equiv_all_3seed/RUN_SUMMARY.md
```

Artifacts:

- `outputs/step2_worker_u_d18_canonical_equiv_all_3seed/results.json`
- `outputs/step2_worker_u_d18_canonical_equiv_all_3seed/SUMMARY.md`
- `outputs/step2_worker_u_d18_canonical_equiv_all_3seed/RUN_SUMMARY.md`

Wall clock: `127.143578` seconds.

## All-Suite Result

| Dataset | D18 final MSE | Best MLP final MSE | MSE diff | MSE W/L/T | D18 test acc | Best MLP test acc | Test acc diff | Test acc W/L/T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `controlled_frequency` | 0.038759 | 0.156857 | +0.118098 | 3/0/0 | | | | |
| `controlled_interaction` | 0.030327 | 0.428272 | +0.397945 | 3/0/0 | | | | |
| `controlled_nonlinear` | 0.017445 | 0.057959 | +0.040513 | 3/0/0 | | | | |
| `controlled_polynomial` | 0.071824 | 0.861150 | +0.788114 | 3/0/0 | | | | |
| `controlled_rare` | 0.018128 | 0.073166 | +0.055038 | 3/0/0 | | | | |
| `controlled_triple` | 0.046849 | 0.605892 | +0.559043 | 3/0/0 | | | | |
| `synthetic_compositional` | 0.240080 | 0.275842 | +0.031623 | 3/0/0 | | | | |
| `synthetic_frequency` | 0.882529 | 1.149273 | +0.265933 | 3/0/0 | | | | |
| `synthetic_polynomial` | 0.798105 | 0.947464 | +0.149360 | 3/0/0 | | | | |
| `digits_iid` | 0.019900 | 0.031013 | +0.011053 | 3/0/0 | 0.968460 | 0.933828 | +0.034014 | 3/0/0 |
| `digits_label_drift` | 0.034849 | 0.038250 | +0.003288 | 3/0/0 | 0.953618 | 0.904143 | +0.046382 | 3/0/0 |
| `digits_mask_noise` | 0.044755 | 0.047837 | +0.003082 | 3/0/0 | 0.827458 | 0.808905 | +0.007421 | 2/1/0 |
| `digits_permuted_pixels` | 0.037868 | 0.049263 | +0.011296 | 3/0/0 | 0.925170 | 0.881262 | +0.043908 | 3/0/0 |
| `digits_class_blocked` | 0.004527 | 0.002992 | -0.001535 | 0/3/0 | 0.218924 | 0.152752 | +0.053185 | 3/0/0 |

All-suite readout: 13/14 final-window MSE rows are positive. The one loss is
`digits_class_blocked`, where the MLP wins current-block final-window MSE but
D18 wins held-out all-class test accuracy. This is the same retention versus
current-block metric ambiguity already documented in D07.

## Code Complexity

The winning candidate is not small.

Line counts:

```text
1426 examples/The Alberta Plan/Step2/new_directions/d07_budgeted_kernel_recursive.py
1454 examples/The Alberta Plan/Step2/new_directions/d10_learned_kernel_resource_manager.py
 691 examples/The Alberta Plan/Step2/new_directions/d14_unified_basis_lms.py
 682 examples/The Alberta Plan/Step2/new_directions/d15_groupwise_basis_lms.py
1654 examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py
5907 total
```

Across those files there are 21 classes and 80 top-level functions. D08 alone
is 1771 lines. The D18 candidate also depends on D07's kernel definitions and
benchmark helpers, D10's learned allocation manager, D14's unified basis, and
D15's groupwise basis learner.

Mechanism count:

- three RKHS core banks with different kernels and allocation schedules;
- learned resource allocation over RKHS banks;
- tanh/Fourier fixed basis readout;
- strict degree-3 polynomial residual RLS;
- unified residual basis;
- learned per-head block gains;
- residual clipping and component clipping.

This is not a compact algorithm in the way LMS, IDBD, Autostep, ObGD, or a
single KRLS learner is compact.

## Critical Assessment

Empirically, D18 canonical-equivalent is the strongest additive/kernel candidate
currently in the repo. It beats the fair MLP grid on the requested weak matrix
with paired 3/0/0 MSE wins on every row, and it also wins the broader 3-seed
suite except for `digits_class_blocked` final-window MSE.

Conceptually, it is only partially satisfactory.

It satisfies the "no router over complete predictors" constraint in the narrow
sense: there is one additive prediction, no MLP expert, and no hard output route
selection. All component blocks update each timestep from the current global
prediction residual.

It does not satisfy a strict "simple universal learner" standard. The success
depends on a hand-assembled set of banks that mirror known benchmark structure:
raw degree-3 polynomial for algebraic tasks, algebraic-Green for retained
geometry, arccosine/tanh/Fourier for composition/frequency, and strict degree-3
RLS for the polynomial blocker. The learned block gains are not a hard router,
but they are a learned combiner over named blocks. That makes the result
stacker-adjacent, even if it is not the forbidden portfolio/router mechanism.

My conclusion:

- Best empirical candidate: D18 canonical-equivalent,
  `d18_gain_safecore_poly_unified_0p01`.
- Best strictly clean additive implementation: D08 multibank single shared
  output matrix, but existing D08 evidence is weaker and not a full closure.
- Honest claim: D18 is a strong additive, non-MLP, non-router empirical closure
  for the current weak matrix. It is not yet a convincing universal principle;
  it is still a carefully engineered feature-bank design.

No source files were edited for this Worker U run.

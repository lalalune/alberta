# Worker W D18 Missing-Regime Assessment

Date: 2026-05-05.

## Audit

D07 `--datasets all` expands to 14 regimes:
`controlled_nonlinear`, `controlled_interaction`, `controlled_triple`,
`controlled_rare`, `controlled_polynomial`, `controlled_frequency`,
`synthetic_polynomial`, `synthetic_frequency`, `synthetic_compositional`,
`digits_iid`, `digits_class_blocked`, `digits_permuted_pixels`,
`digits_mask_noise`, and `digits_label_drift`.

Existing D18 canonical evidence in
`outputs/step2_new_directions/d18_gain_safecore_poly_unified_clip1_eta02_l2_0p05_broad_10seed`
covers 8/14 regimes: `controlled_frequency`, `controlled_interaction`,
`digits_label_drift`, `digits_mask_noise`, `digits_permuted_pixels`,
`synthetic_compositional`, `synthetic_frequency`, and `synthetic_polynomial`.
The missing regimes were `controlled_nonlinear`, `controlled_triple`,
`controlled_rare`, `controlled_polynomial`, `digits_iid`, and
`digits_class_blocked`.

## Commands

Initial missing-regime run:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" --configs step2_canonical --datasets controlled_nonlinear,controlled_triple,controlled_rare,controlled_polynomial,digits_iid,digits_class_blocked --n-seeds 3 --seed 0 --steps 1200 --final-window 300 --output-dir outputs/step2_new_directions/worker_w_d18_missing_3seed --note-path docs/research/step2_new_directions/worker_w_d18_missing_3seed.md
```

Failing-regime scalar sweeps:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" --configs step2_canonical --datasets digits_class_blocked --n-seeds 3 --seed 0 --steps 1200 --final-window 300 --basis-weight-decay 0.995 --output-dir outputs/step2_new_directions/worker_w_d18_class_blocked_basisdecay_0p995_3seed --note-path docs/research/step2_new_directions/worker_w_d18_class_blocked_basisdecay_0p995_3seed.md
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" --configs step2_canonical --datasets digits_class_blocked --n-seeds 3 --seed 0 --steps 1200 --final-window 300 --basis-weight-decay 0.99 --output-dir outputs/step2_new_directions/worker_w_d18_class_blocked_basisdecay_0p99_3seed --note-path docs/research/step2_new_directions/worker_w_d18_class_blocked_basisdecay_0p99_3seed.md
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" --configs step2_canonical --datasets digits_class_blocked --n-seeds 3 --seed 0 --steps 1200 --final-window 300 --unified-step-size 0.2 --output-dir outputs/step2_new_directions/worker_w_d18_class_blocked_unifiedstep_0p2_3seed --note-path docs/research/step2_new_directions/worker_w_d18_class_blocked_unifiedstep_0p2_3seed.md
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" --configs step2_canonical --datasets digits_class_blocked --n-seeds 3 --seed 0 --steps 1200 --final-window 300 --unified-step-size 0.6 --output-dir outputs/step2_new_directions/worker_w_d18_class_blocked_unifiedstep_0p6_3seed --note-path docs/research/step2_new_directions/worker_w_d18_class_blocked_unifiedstep_0p6_3seed.md
```

The 10-seed missing-regime confirmation was not run because the 3-seed
canonical pass had a negative primary-metric result.

## Existing Broad 10-Seed Evidence

Positive `final_window_mse` differences favor D18 over the best fair MLP.

| Regime | MSE diff | SE | W/L/T | Test accuracy diff |
|---|---:|---:|---:|---:|
| controlled_frequency | +0.1438 | 0.0158 | 10/0/0 |  |
| controlled_interaction | +0.4522 | 0.0303 | 10/0/0 |  |
| digits_label_drift | +0.0044 | 0.0006 | 10/0/0 | +0.0458 |
| digits_mask_noise | +0.0058 | 0.0009 | 10/0/0 | +0.0080 |
| digits_permuted_pixels | +0.0107 | 0.0004 | 10/0/0 | +0.0442 |
| synthetic_compositional | +0.0449 | 0.0187 | 9/1/0 |  |
| synthetic_frequency | +0.5345 | 0.1240 | 10/0/0 |  |
| synthetic_polynomial | +0.1485 | 0.0353 | 10/0/0 |  |

## Missing-Regime 3-Seed Results

| Regime | MSE diff | SE | W/L/T | Test accuracy diff |
|---|---:|---:|---:|---:|
| controlled_nonlinear | +0.0405 | 0.0031 | 3/0/0 |  |
| controlled_polynomial | +0.7881 | 0.1669 | 3/0/0 |  |
| controlled_rare | +0.0550 | 0.0056 | 3/0/0 |  |
| controlled_triple | +0.5590 | 0.0321 | 3/0/0 |  |
| digits_class_blocked | -0.0015 | 0.0001 | 0/3/0 | +0.0532 |
| digits_iid | +0.0111 | 0.0005 | 3/0/0 | +0.0340 |

## Failing-Regime Sweep

| Setting | MSE diff | SE | W/L/T | Test accuracy diff | Acc W/L/T |
|---|---:|---:|---:|---:|---:|
| step2_canonical | -0.0015 | 0.0001 | 0/3/0 | +0.0532 | 3/0/0 |
| basis_weight_decay=0.995 | -0.0013 | 0.0001 | 0/3/0 | +0.0458 | 3/0/0 |
| basis_weight_decay=0.99 | -0.0012 | 0.0002 | 0/3/0 | +0.0550 | 3/0/0 |
| unified_step_size=0.2 | -0.0015 | 0.0001 | 0/3/0 | +0.0550 | 3/0/0 |
| unified_step_size=0.6 | -0.0015 | 0.0001 | 0/3/0 | +0.0507 | 3/0/0 |

`basis_weight_decay=0.99` is the best sweep result by primary MSE, but it is
still negative on all three paired seeds.

## Assessment

D18 now has coverage evidence for all 14 D07 `all` regimes, but it does not
have full 14-regime positive primary-metric evidence. The current fixed
`step2_canonical` configuration is positive on 13/14 regimes by final-window
MSE and negative on `digits_class_blocked`, despite being positive on held-out
test accuracy for that same regime. Under the stated protocol, this blocks a
10-seed confirmation run on the missing set.

The best result is the existing 10-seed `controlled_frequency`/`controlled_interaction`
and synthetic evidence plus the new 3-seed missing-regime run, with the largest
new missing-regime margin on `controlled_polynomial` (+0.7881 final-window MSE
diff versus best fair MLP). The only unresolved blocker is
`digits_class_blocked` final-window MSE.

Constraint check: D18 does not use an output router, a discrete portfolio, an
MLP expert selector, or hand-engineered per-regime selection in these runs. It
does, however, learn internal component gains over additive resource-basis
blocks. I would treat that as internal information sharing only if the final
learner is defined as one additive model with all blocks updated at every time
step. If the user constraint is interpreted strictly as forbidding learned
output weighting across component families, D18 is too close to a portfolio
blend and should not be promoted without simplifying those gains into ordinary
model parameters or fixing them globally.

Conclusion: D18 is not promotable as the current simple Step 2 candidate on
primary final-window MSE. It is the strongest simple-looking candidate in this
audit, but it remains blocked by `digits_class_blocked` unless the acceptance
criterion is changed to prioritize classification accuracy for digits regimes.

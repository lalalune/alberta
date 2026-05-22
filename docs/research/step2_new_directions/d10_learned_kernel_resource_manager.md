# D10 Learned Kernel Resource Manager

## Purpose

D10 tests a learned resource manager for the D07 kernel/RKHS direction.  The
core constraint is that this must not become a prediction router: prediction is
always the sum of active banks,
`y_hat = y_raw_poly + y_algebraic_green + y_arccosine`, and every bank updates
its existing coefficients on every time step under one global loss.  The
manager only decides where scarce representation resources are spent: center
birth order, optional global-budget rebalancing, and optional per-bank RLS
forgetting.

## Implemented Algorithm

The implemented learner lives in
`examples/The Alberta Plan/Step2/new_directions/d10_learned_kernel_resource_manager.py`.

The three additive banks are:

| Bank | Role | Promoted setting |
|---|---|---:|
| raw polynomial | exact low-dimensional polynomial and frequency structure | degree 3, raw normalization, budget 64 |
| algebraic-Green | D07 stateful external-memory bank | algebraic weight 0.75, budget 128, add interval 8 |
| arc-cosine / NNGP | smooth compositional nonlinear bank | depth 1, budget 128, add interval 2 |

At each step the learner:

1. predicts with all banks and sums their outputs;
2. computes global prequential loss;
3. computes ALD novelty for every bank;
4. estimates per-bank allocation utility from residual loss, novelty, budget
   pressure, and configured compute cost;
5. lets the manager pick at most one bank to receive a center opportunity;
6. updates all existing bank coefficients against the additive residual;
7. updates manager preferences from utility and same-sample loss improvement.

The manager variants are:

| Variant | What It Tests |
|---|---|
| `learned_softmax` | learned cost-sensitive exponentiated-gradient allocation |
| `novelty_greedy` | non-learned ALD novelty allocation ablation |
| `round_robin` | non-learned fixed allocation ablation |

Two risk controls were added after ablation:

| Mechanism | Result |
|---|---|
| Cross-bank center transfers | Implemented, but disabled by default.  It caused destructive churn because dropping a center resets that bank covariance. |
| Learned rho / forgetting allocation | Implemented, but defaulted to 0 span.  It hurt the learned manager on mask noise. |

## Completed Runs

All runs used 1200 online steps, a 300-step final window, paired seeds unless
noted, and fair MLP baselines `mlp_h64`, `mlp_h128`, and `mlp_h64_64`.

| Run | Path | Purpose |
|---|---|---|
| Shared-budget destructive-transfer probe | `outputs/step2_new_directions/d10_blocker_probe_1seed/results.json` | Tests learned cross-bank center transfers under a 160-center shared budget. |
| 192-center shared budget | `outputs/step2_new_directions/d10_resource_manager_blockers_3seed/results.json` | Tests true scarcity without transfers. |
| D07 algebraic full suite | `outputs/step2_new_directions/d10_resource_manager_d07_algebraic_3seed/results.json` | Full six-dataset blocker suite with D07-proven algebraic settings and rho allocation enabled. |
| No-rho failure isolation | `outputs/step2_new_directions/d10_failure_modes_no_rho_3seed/results.json` | Isolates the two remaining failures with learned rho disabled. |

## Main Full-Suite Result

The best completed full-suite result is
`outputs/step2_new_directions/d10_resource_manager_d07_algebraic_3seed/results.json`.
Positive differences favor the manager over the per-seed best fair MLP.

| Dataset | Primary Metric | Best Manager Diff | Best W/L/T | Learned Diff | Learned W/L/T | Verdict |
|---|---:|---:|---:|---:|---:|---|
| controlled_frequency | final-window MSE | +0.1456 | 3/0/0 | +0.1456 | 3/0/0 | solved |
| controlled_nonlinear | final-window MSE | +0.0427 | 3/0/0 | +0.0409 | 3/0/0 | solved |
| digits_label_drift | test accuracy | +0.0544 | 3/0/0 | +0.0451 | 3/0/0 | solved |
| digits_permuted_pixels | test accuracy | +0.0464 | 3/0/0 | +0.0247 | 3/0/0 | solved |
| digits_mask_noise | test accuracy | +0.0223 | 3/0/0 | -0.0148 | 1/2/0 | ablation solved, learned rho hurt |
| synthetic_compositional | final-window MSE | -0.0524 | 0/3/0 | -0.0952 | 0/3/0 | not solved |

The no-rho isolation run fixes the learned manager on mask noise:

| Dataset | Primary Metric | Best Manager Diff | Best W/L/T | Learned Diff | Learned W/L/T |
|---|---:|---:|---:|---:|---:|
| digits_mask_noise | test accuracy | +0.0229 | 3/0/0 | +0.0204 | 3/0/0 |
| synthetic_compositional | final-window MSE | -0.0493 | 0/3/0 | -0.0835 | 0/3/0 |

Therefore the promoted code defaults disable learned rho allocation.  That
keeps the learned manager competitive on mask noise, but it does not solve the
synthetic compositional blocker.

## Ablation Findings

Cross-bank center transfers are not viable in the current implementation.  In
the 1-seed transfer probe, controlled_frequency flipped from a large win to a
large loss: best-manager final-window MSE difference was `-2.6316`.  The reason
is mechanical: transferring a center deletes an active basis element and resets
the donor bank covariance.  With hundreds of transfers, the learner churns
instead of retaining useful structure.

A smaller 192-center shared budget is useful for frequency and several digit
regimes, but it starves the algebraic memory bank.  It wins controlled_frequency
`+0.1460`, controlled_nonlinear `+0.0433`, label drift `+0.0519` test accuracy,
and permuted pixels `+0.0254` test accuracy, but loses mask noise `-0.0618` test
accuracy and compositional `-0.1247` final-window MSE.

The D07 algebraic settings matter.  Raising the algebraic bank to 128 centers
and algebraic weight 0.75 restores strong digit behavior, especially mask noise
for the non-learned allocation ablations.  This confirms that D10 should inherit
D07's stateful-memory configuration rather than retune it downward for global
budget pressure.

Learned rho allocation is harmful.  With rho allocation enabled, learned
softmax loses mask-noise test accuracy despite the best manager winning.  With
rho span set to 0, learned softmax wins mask-noise test accuracy on all three
seeds.  The code keeps rho allocation available, but the default is now off.

The arc-cosine bank is still not enough for synthetic compositional.  Increasing
arc-cosine budget/depth in probes did not beat MLP.  The best manager narrowed
the compositional gap from roughly `-0.1247` under the 192-center budget to
`-0.0493` with D07 algebraic settings and no rho allocation, but it still lost
on all three paired seeds.

## Compute

The D10 additive kernel learner is materially more expensive than the fair MLP
baselines in these Python/NumPy research runners.  Runtime ratios from the
generated summaries are typically:

| Regime | D10 Runtime Pattern |
|---|---|
| controlled tasks | about 4x to 9x fastest MLP, depending on manager and dataset |
| synthetic_compositional | about 2.7x to 4.5x fastest MLP |
| digits | about 3x to 6x fastest MLP in the completed runs |

This is expected: each bank evaluates a growing dictionary and RLS covariance,
while MLP baselines run JAX-compiled scans.  The D10 code is a research probe,
not an optimized kernel implementation.

## Assessment

D10 implements the missing learned resource-manager mechanism without turning
the learner into a prediction router.  It proves that resource allocation over
D07-style RKHS banks can beat fair MLP on several important blockers, including
controlled_frequency, controlled_nonlinear, digits_label_drift, digits_mask_noise
after the no-rho fix, and digits_permuted_pixels.

It does not close Step 2 beyond doubt.  The remaining blocker is
`synthetic_compositional`: best-manager final-window MSE remains worse than the
best fair MLP by about `0.0493` to `0.0524` across the strongest completed D10
runs, with losses on all three paired seeds.  The current arc-cosine bank does
not supply the missing recursive compositional representation.  The next
mechanism should not be more routing; it should be a genuinely stronger
compositional basis or recursive feature-construction bank that can replace the
arc-cosine bank inside the same additive/resource-managed framework.

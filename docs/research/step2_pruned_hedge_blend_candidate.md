# Step 2 Pruned Discounted-Hedge Blend Candidate

Superseded: see `docs/research/step2_stacker006_pruned_candidate.md`.
The newer candidate adds the low-rate prediction stacker, uses
`--selector-window 300`, and changes held-out digit blending to
`--h128-blend-weight 0.4`.

Candidate profile:

- Routing/deployment excludes `upgd_low_noise` and `dynamic_sparse`.
- Convex routes use cumulative discounted Hedge: `--weighting-scheme discounted_hedge --hedge-eta 1.0 --hedge-discount 0.995`.
- Selector window is `500`.
- Held-out digits deployment uses `--digits-deployment-objective all_h128_blend --h128-blend-weight 0.5`.
- Recursive, polynomial, Fourier, random tanh, and the full fair MLP grid remain available; the aggressive prune that removed small MLPs/tanh regressed controlled nonlinear and synthetic compositional tasks.

## Results

Positive deltas favor conclusive over the same-run best fair MLP.

### digits

Artifact: `outputs/step2_conclusive_hedge_blend_no_upgd_dynamic_digits_3seed/results.json`

| Dataset | Final MSE diff | Final W/L/T | Test MSE diff | Test Acc diff | Test Acc W/L/T |
|---|---:|---:|---:|---:|---:|
| `digits_class_blocked` | +0.000000 | 0/0/3 | +0.043431 | +0.366728 | 3/0/0 |
| `digits_iid` | +0.006568 | 3/0/0 | +0.002490 | +0.011750 | 3/0/0 |
| `digits_label_drift` | +0.007664 | 3/0/0 | +0.002597 | +0.004329 | 3/0/0 |
| `digits_mask_noise` | +0.006948 | 3/0/0 | +0.003017 | +0.015461 | 3/0/0 |
| `digits_permuted_pixels` | +0.008283 | 3/0/0 | +0.005220 | +0.012369 | 2/1/0 |

### controlled_synth

Artifact: `outputs/step2_conclusive_hedge_blend_no_upgd_dynamic_controlled_synth_3seed/results.json`

| Dataset | Final MSE diff | Final W/L/T | Test MSE diff | Test Acc diff | Test Acc W/L/T |
|---|---:|---:|---:|---:|---:|
| `controlled_frequency` | +0.175881 | 3/0/0 |  |  |  |
| `controlled_interaction` | +0.136030 | 3/0/0 |  |  |  |
| `controlled_nonlinear` | +0.010569 | 3/0/0 |  |  |  |
| `controlled_polynomial` | +0.530434 | 3/0/0 |  |  |  |
| `controlled_rare` | +0.009967 | 3/0/0 |  |  |  |
| `controlled_triple` | +0.460076 | 3/0/0 |  |  |  |
| `synthetic_compositional` | +0.048063 | 3/0/0 |  |  |  |
| `synthetic_frequency` | +0.422978 | 3/0/0 |  |  |  |
| `synthetic_polynomial` | +0.125385 | 1/2/0 |  |  |  |

## Assessment

This candidate fixes the previous `digits_label_drift` held-out accuracy gap: over 3 seeds, test accuracy is `+0.004329` against the per-seed best fair MLP with `3/0/0` wins. The old full tuned conclusive run was `-0.003711` with `1/2/0` wins.

The lighter prune is materially better than the hard prune. Removing UPGD and dynamic sparse preserves every controlled/synthetic mean final-window MSE advantage. Removing the small MLP controls and random tanh in addition caused controlled nonlinear and synthetic compositional regressions, so those components should stay in the canonical candidate for now.

The remaining weak spot is `synthetic_polynomial`: mean final-window MSE remains positive (`+0.125385`), but seed wins/losses/ties are `1/2/0`, matching the earlier caveat pattern. This is not a regression from the prior conclusive result, but it is still the next target for stronger seed-level robustness.

Cost interpretation: this prunes UPGD/dynamic from conclusive routing and deployment. The current runner still executes disabled experts for paired diagnostics, so this is an algorithmic prune, not yet a physical wall-clock prune. A physically pruned runner should remove those learner states entirely once the candidate is frozen.

# Worker C D18 Complexity Pruning Report

Scope: analysis only. This report reads existing result artifacts plus the optional Worker C tanh64 ablation when present. It does not import or modify D18.

## All14 Primary Final-Window MSE

| run | seeds | datasets | mean margin | worst dataset | worst margin | worst W/L/T | all W/L/T | neg cells | runtime vs fastest MLP | wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical full budget | 10 | 14 | 0.212603 | digits_class_blocked | 0.001277 | 10/0/0 | 140/0/0 | 0 | 8.43x mean / 12.08x max | 374.7 |
| full budget tanh128 | 10 | 14 | 0.212203 | digits_class_blocked | 0.001277 | 10/0/0 | 140/0/0 | 0 | 8.38x mean / 10.80x max | 326.5 |
| half budget tanh128 | 10 | 14 | 0.198222 | digits_class_blocked | 0.001277 | 10/0/0 | 139/1/0 | 0 | 9.30x mean / 14.86x max | 894.6 |
| quarter budget tanh256 | 3 | 14 | 0.136344 | digits_mask_noise | -0.001859 | 1/2/0 | 37/5/0 | 2 | 8.46x mean / 11.83x max | 237.8 |
| full budget tanh64 | 3 | 14 | 0.177393 | digits_class_blocked | 0.001437 | 3/0/0 | 42/0/0 | 0 | 12.92x mean / 21.91x max | 801.5 |

Margins are paired best-MLP MSE minus candidate MSE, so positive values favor D18.

## Digit Heldout Cells

| run | metric cells | neg cells | worst cell | worst margin | worst W/L/T | mean heldout margin |
| --- | --- | --- | --- | --- | --- | --- |
| canonical full budget | 10 | 0 | digits_mask_noise:test_mse | 0.003226 | 7/3/0 | 0.097335 |
| full budget tanh128 | 10 | 0 | digits_mask_noise:test_mse | 0.003085 | 7/3/0 | 0.097569 |
| half budget tanh128 | 10 | 1 | digits_mask_noise:test_accuracy | -0.005380 | 6/4/0 | 0.090722 |
| quarter budget tanh256 | 10 | 3 | digits_mask_noise:test_accuracy | -0.025974 | 0/3/0 | 0.079501 |
| full budget tanh64 | 10 | 0 | digits_mask_noise:test_mse | 0.000073 | 1/2/0 | 0.095242 |

Digit heldout cells include test MSE and test accuracy for the five digit regimes. For MSE, positive means lower MSE than best MLP; for accuracy, positive means higher accuracy than best MLP.

## Available No-Poly / No-Unified Evidence

| run | method | datasets | mean final MSE margin | worst margin | worst W/L/T | runtime vs fastest MLP |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic compositional canonical | d18_step2_canonical | synthetic_compositional | 0.044882 | 0.044882 | 9/1/0 | 5.94x |
| synthetic compositional no unified | d18_step2_no_unified | synthetic_compositional | 0.037011 | 0.037011 | 8/2/0 | 6.10x |
| synthetic compositional no poly | d18_step2_no_poly | synthetic_compositional | 0.040113 | 0.040113 | 9/1/0 | 6.13x |

The no-poly and no-unified artifacts are only available on synthetic_compositional; they are not all14 evidence.

## Existing OPMNIST Evidence

| run | method | steps | blocks | core protocol | published task count | final MSE margin | final acc margin | test MSE margin | test acc margin | elapsed s | steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OPMNIST 31 full task blocks | mixture | 1860000 | 31 | True | False | 0.003018 | 0.008533 | 0.007594 | 0.047477 | 5265.0 | 353.3 |
| OPMNIST 40 full task blocks | mixture | 2400000 | 40 | True | False | 0.002250 | 0.003550 | 0.019942 | 0.011013 | 6624.3 | 362.3 |

This OPMNIST evidence is for the existing strict Step 2 portfolio, not D18. It therefore proves that the project has a partial true-MNIST OPMNIST runner and a positive non-D18 result, but it does not yet prove the simplified D18 candidate on OPMNIST.

## Recommendation

Promote full-budget tanh64 to a 10-seed all14 confirmation run. It is the smallest passing candidate in the available table, but its evidence is only 3 seeds until that run is repeated.

Engineering conclusion: prune the random tanh basis first, then confirm at 10 seeds. Do not prune the RKHS bank budgets until a learned allocator or replacement mechanism recovers the half/quarter-budget losses. The next production track should be a fused JAX implementation of the smallest confirmed candidate, with D18 wired into the OPMNIST runner before making external-benchmark claims.

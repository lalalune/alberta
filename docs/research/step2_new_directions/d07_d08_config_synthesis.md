# D07/D08 Configuration Synthesis for Step 2

## Scope

This document synthesizes the Step 2 new-direction outputs currently present
under `outputs/step2_new_directions/*/results.json`. There are 12 result files,
all from D07 runs. No `d08*` output directory is present at the time of this
readout, so this is a D07 evidence synthesis plus an explicit specification of
what D08 must resolve.

The extraction is from the JSON `aggregate` blocks, not from memory. For
regression benchmarks the primary metric is `final_window_mse`, lower is better.
For digit benchmarks the primary deployment metric is held-out `test_accuracy`,
higher is better, with `test_mse` and final-window metrics reported separately.
This distinction matters most for `digits_class_blocked`: final-window stream
metrics reward specialization to the current class block, while held-out metrics
measure retention across all classes.

Positive deltas favor the D07 candidate over the best MLP. `W/L/T` is paired
seed wins, losses, and ties against the best MLP in the same run and metric.

## Result Files Read

| Run | Seeds | Steps | Wall clock s | Datasets |
|---|---:|---:|---:|---|
| `d07_algebraic_all_3seed_1200` | 3 | 1200 | 346.4 | controlled suite, synthetic suite, digits suite |
| `d07_algebraic_controlled_2seed` | 2 | 600 | 21.2 | controlled triple/polynomial/frequency |
| `d07_arccosine_comp_b256_3seed` | 3 | 1200 | 33.4 | synthetic compositional |
| `d07_arccosine_comp_depth1_3seed` | 3 | 1200 | 97.9 | synthetic compositional |
| `d07_budgeted_kernel_recursive` | 1 | 120 | 2.3 | controlled nonlinear smoke |
| `d07_digits_permute_mask_interval_3seed` | 3 | 1200 | 166.4 | digits permuted pixels, digits mask noise |
| `d07_hybrid_probe_3seed` | 3 | 1200 | 192.1 | controlled frequency/triple, synthetic compositional, stateful digits |
| `d07_hybrid_rawpoly_freq_comp_3seed` | 3 | 1200 | 41.5 | controlled frequency, synthetic compositional |
| `d07_label_drift_interval_3seed` | 3 | 1200 | 290.7 | digits label drift |
| `d07_raw_algebraic_green_probe_3seed` | 3 | 1200 | 71.4 | controlled frequency, synthetic compositional, stateful digits |
| `d07_rawpoly_remaining_3seed` | 3 | 1200 | 190.4 | controlled frequency, synthetic frequency, synthetic compositional |
| `d07_sweep_controlled_2seed` | 2 | 600 | 47.8 | controlled suite |

## Best D07 Candidate Per Benchmark

This table selects, for each benchmark, the strongest 3-seed 1200-step
D07 candidate found in the JSONs, and compares it to the best fair MLP in the
same result file. This is deliberately a best-available table, not a claim that
one fixed learner has won all benchmarks. Most winning rows are pure kernel
learners. The `synthetic_compositional` row is an MLP-containing residual hybrid
and is included only because it is the best observed D07 candidate there; it is
not admissible for a final non-MLP learner claim.

| Benchmark | Primary metric | Best D07 candidate config | Candidate mean | Best MLP | MLP mean | Delta | W/L/T | Runtime ratio | Centers | Source |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `controlled_frequency` | `final_window_mse` | `polynomial_rls_b128_s1_r0p99_n0p001_d3_aw0p5_rawpoly` | 0.0030 | `mlp_h64_64` | 0.1569 | +0.1538 | 3/0/0 | 0.67x | 128.0 | `d07_hybrid_rawpoly_freq_comp_3seed` |
| `controlled_interaction` | `final_window_mse` | `algebraic_green_rls_b128_s0p5_r1_n0p05_d3_aw0p75` | 0.2001 | `mlp_h64` | 0.4283 | +0.2282 | 3/0/0 | 2.13x | 128.0 | `d07_algebraic_all_3seed_1200` |
| `controlled_nonlinear` | `final_window_mse` | `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.0308 | `mlp_h64` | 0.0580 | +0.0271 | 3/0/0 | 1.16x | 128.0 | `d07_algebraic_all_3seed_1200` |
| `controlled_polynomial` | `final_window_mse` | `algebraic_green_rls_b128_s1_r1_n0p001_d3_aw0p75` | 0.3524 | `mlp_h64` | 0.8611 | +0.5088 | 3/0/0 | 3.63x | 128.0 | `d07_algebraic_all_3seed_1200` |
| `controlled_rare` | `final_window_mse` | `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.0447 | `mlp_h64` | 0.0732 | +0.0284 | 3/0/0 | 1.86x | 128.0 | `d07_algebraic_all_3seed_1200` |
| `controlled_triple` | `final_window_mse` | `algebraic_green_rls_b128_s1_r1_n0p05_d3_aw0p75` | 0.2137 | `mlp_h64_64` | 0.6059 | +0.3922 | 3/0/0 | 3.22x | 128.0 | `d07_algebraic_all_3seed_1200` |
| `digits_class_blocked` | `test_accuracy` | `algebraic_green_rls_b128_s1_r1_n0p05_d3_aw0p75` | 0.8454 | `mlp_h128` | 0.1528 | +0.6926 | 3/0/0 | 4.16x | 128.0 | `d07_algebraic_all_3seed_1200` |
| `digits_iid` | `test_accuracy` | `algebraic_green_rls_b128_s0p5_r1_n0p05_d3_aw0p75` | 0.9647 | `mlp_h128` | 0.9338 | +0.0309 | 3/0/0 | 0.77x | 128.0 | `d07_algebraic_all_3seed_1200` |
| `digits_label_drift` | `test_accuracy` | `algebraic_green_rls_b128_s1_r0p995_n0p001_d3_aw0p75_ai4` | 0.9468 | `mlp_h128` | 0.9041 | +0.0427 | 3/0/0 | 0.36x | 128.0 | `d07_label_drift_interval_3seed` |
| `digits_mask_noise` | `test_accuracy` | `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_rawpoly_ai8` | 0.8677 | `mlp_h64` | 0.8089 | +0.0588 | 3/0/0 | 1.33x | 128.0 | `d07_raw_algebraic_green_probe_3seed` |
| `digits_permuted_pixels` | `test_accuracy` | `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_rawpoly_ai8` | 0.9233 | `mlp_h128` | 0.8813 | +0.0421 | 3/0/0 | 1.14x | 128.0 | `d07_raw_algebraic_green_probe_3seed` |
| `synthetic_compositional` | `final_window_mse` | `hybrid_mlp_h128_algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai8` | 0.3050 | `mlp_h128` | 0.2758 | -0.0292 | 0/3/0 | 4.22x | 128.0 | `d07_hybrid_probe_3seed` |
| `synthetic_frequency` | `final_window_mse` | `polynomial_rls_b128_s1_r0p99_n0p05_d3_aw0p5_rawpoly_ai4` | 1.1590 | `mlp_h64_64` | 1.1493 | -0.0097 | 2/1/0 | 0.14x | 45.3 | `d07_rawpoly_remaining_3seed` |
| `synthetic_polynomial` | `final_window_mse` | `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.9314 | `mlp_h64_64` | 0.9475 | +0.0161 | 2/1/0 | 1.53x | 128.0 | `d07_algebraic_all_3seed_1200` |

Immediate readout:

- D07 has a non-MLP win available on 12 of 14 primary rows if held-out digit
  metrics are used.
- It does not have a non-MLP win on `synthetic_compositional`.
- The best `synthetic_compositional` row contains an MLP residual hybrid and
  still loses; the best pure kernel row is 0.3725, also a loss.
- It barely misses `synthetic_frequency` on mean final-window MSE despite
  winning two of three paired seeds.
- The table is not sufficient for a universal learner claim because it uses
  different configurations across benchmarks.

## Digit Secondary Metrics

The digit metrics expose a critical evaluation ambiguity. On class-blocked
training, the MLP dominates the final stream block because it specializes to
the currently visible classes. The kernel learner dominates held-out all-class
retention.

| Benchmark | Metric | Best non-MLP | Non | Best MLP | MLP | Delta | W/L/T | Source |
|---|---|---|---:|---|---:|---:|---:|---|
| `digits_class_blocked` | `final_window_mse` | `algebraic_green_rls_b128_s1_r1_n0p001_d3_aw0p75` | 0.0583 | `mlp_h64_64` | 0.0030 | -0.0553 | 0/3/0 | `d07_algebraic_all_3seed_1200` |
| `digits_class_blocked` | `final_window_accuracy` | `algebraic_green_rls_b128_s1_r1_n0p05_d3_aw0p75` | 0.6456 | `mlp_h64_64` | 0.9922 | -0.3467 | 0/3/0 | `d07_algebraic_all_3seed_1200` |
| `digits_class_blocked` | `test_mse` | `algebraic_green_rls_b128_s1_r1_n0p001_d3_aw0p75` | 0.0438 | `mlp_h128` | 0.1343 | +0.0905 | 3/0/0 | `d07_algebraic_all_3seed_1200` |
| `digits_class_blocked` | `test_accuracy` | `algebraic_green_rls_b128_s1_r1_n0p05_d3_aw0p75` | 0.8454 | `mlp_h128` | 0.1528 | +0.6926 | 3/0/0 | `d07_algebraic_all_3seed_1200` |
| `digits_iid` | `final_window_mse` | `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.0188 | `mlp_h64` | 0.0310 | +0.0122 | 3/0/0 | `d07_algebraic_all_3seed_1200` |
| `digits_iid` | `final_window_accuracy` | `algebraic_green_rls_b128_s1_r1_n0p05_d3_aw0p75` | 0.9511 | `mlp_h64` | 0.9133 | +0.0378 | 3/0/0 | `d07_algebraic_all_3seed_1200` |
| `digits_iid` | `test_mse` | `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.0176 | `mlp_h128` | 0.0287 | +0.0111 | 3/0/0 | `d07_algebraic_all_3seed_1200` |
| `digits_iid` | `test_accuracy` | `algebraic_green_rls_b128_s0p5_r1_n0p05_d3_aw0p75` | 0.9647 | `mlp_h128` | 0.9338 | +0.0309 | 3/0/0 | `d07_algebraic_all_3seed_1200` |
| `digits_label_drift` | `final_window_mse` | `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai8` | 0.0321 | `mlp_h64` | 0.0383 | +0.0062 | 3/0/0 | `d07_hybrid_probe_3seed` |
| `digits_label_drift` | `final_window_accuracy` | `algebraic_green_rls_b128_s1_r0p99_n0p001_d3_aw0p75_ai8` | 0.8878 | `mlp_h64` | 0.8600 | +0.0278 | 3/0/0 | `d07_label_drift_interval_3seed` |
| `digits_label_drift` | `test_mse` | `algebraic_green_rls_b128_s0p5_r0p995_n0p001_d3_aw0p75_ai8` | 0.0227 | `mlp_h64` | 0.0342 | +0.0115 | 3/0/0 | `d07_label_drift_interval_3seed` |
| `digits_label_drift` | `test_accuracy` | `algebraic_green_rls_b128_s1_r0p995_n0p001_d3_aw0p75_ai4` | 0.9468 | `mlp_h128` | 0.9041 | +0.0427 | 3/0/0 | `d07_label_drift_interval_3seed` |
| `digits_mask_noise` | `final_window_mse` | `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai4` | 0.0443 | `mlp_h128` | 0.0478 | +0.0035 | 2/1/0 | `d07_digits_permute_mask_interval_3seed` |
| `digits_mask_noise` | `final_window_accuracy` | `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_rawpoly_ai8` | 0.8033 | `mlp_h128` | 0.8067 | -0.0033 | 1/1/1 | `d07_raw_algebraic_green_probe_3seed` |
| `digits_mask_noise` | `test_mse` | `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai8` | 0.0366 | `mlp_h128` | 0.0447 | +0.0081 | 3/0/0 | `d07_digits_permute_mask_interval_3seed` |
| `digits_mask_noise` | `test_accuracy` | `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_rawpoly_ai8` | 0.8677 | `mlp_h64` | 0.8089 | +0.0588 | 3/0/0 | `d07_raw_algebraic_green_probe_3seed` |
| `digits_permuted_pixels` | `final_window_mse` | `algebraic_green_rls_b128_s1_r0p99_n0p001_d3_aw0p75_ai4` | 0.0359 | `mlp_h128` | 0.0493 | +0.0134 | 3/0/0 | `d07_digits_permute_mask_interval_3seed` |
| `digits_permuted_pixels` | `final_window_accuracy` | `algebraic_green_rls_b128_s1_r0p99_n0p001_d3_aw0p75_ai4` | 0.8956 | `mlp_h128` | 0.8100 | +0.0856 | 3/0/0 | `d07_digits_permute_mask_interval_3seed` |
| `digits_permuted_pixels` | `test_mse` | `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai8` | 0.0298 | `mlp_h128` | 0.0397 | +0.0099 | 3/0/0 | `d07_digits_permute_mask_interval_3seed` |
| `digits_permuted_pixels` | `test_accuracy` | `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_rawpoly_ai8` | 0.9233 | `mlp_h128` | 0.8813 | +0.0421 | 3/0/0 | `d07_raw_algebraic_green_probe_3seed` |

## What Each Candidate Solves and Breaks

### Normalized algebraic-Green, no throttling

Representative config:

```text
algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75
```

This is the strongest broad non-MLP family. It wins:

- `controlled_nonlinear`: 0.0308 vs MLP 0.0580, delta +0.0271.
- `controlled_interaction`: 0.2015 vs MLP 0.4283, delta +0.2268.
- `controlled_polynomial`: 0.3824 vs MLP 0.8611, delta +0.4788.
- `controlled_rare`: 0.0447 vs MLP 0.0732, delta +0.0284.
- `controlled_triple`: 0.2410 vs MLP 0.6059, delta +0.3649.
- `synthetic_polynomial`: 0.9314 vs MLP 0.9475, delta +0.0161.
- `digits_iid` held-out accuracy: 0.9647 vs MLP 0.9338, delta +0.0309.
- `digits_class_blocked` held-out accuracy: 0.8417 vs MLP 0.1528, delta +0.6889.

It breaks:

- `controlled_frequency`: 0.3048 vs MLP 0.1569, delta -0.1479.
- `synthetic_frequency`: 1.3475 vs MLP 1.1493, delta -0.1982.
- `synthetic_compositional`: 1.0813 vs MLP 0.2758, delta -0.8055.
- `digits_label_drift` held-out accuracy: 0.4218 vs MLP 0.9041, delta -0.4824.
- `digits_mask_noise` held-out accuracy: 0.7242 vs MLP 0.8089, delta -0.0847.
- `digits_permuted_pixels` held-out accuracy: 0.7205 vs MLP 0.8813, delta -0.1608.

Conclusion: this is an excellent algebraic/retention learner, but it spends its
dictionary too early for stateful digits and lacks the right bias for frequency
and tanh composition.

### Throttled algebraic-Green

Representative configs:

```text
algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai8
algebraic_green_rls_b128_s1_r0p995_n0p001_d3_aw0p75_ai4
```

Throttling fixes stateful digits by reserving center capacity for later phases.
The best held-out test accuracy deltas are:

- `digits_label_drift`: 0.9468 vs MLP 0.9041, delta +0.0427.
- `digits_mask_noise`: 0.8677 vs MLP 0.8089, delta +0.0588.
- `digits_permuted_pixels`: 0.9233 vs MLP 0.8813, delta +0.0421.

It still breaks:

- `controlled_frequency`: 0.2351 vs MLP 0.1569, delta -0.0783.
- `synthetic_compositional`: 0.3061 vs MLP 0.2758, delta -0.0303.

Conclusion: center scheduling is a real resource-management mechanism, not a
cosmetic hyperparameter. It is necessary for the stateful external benchmarks
but insufficient for the remaining compositional blocker.

### Raw degree-3 polynomial RKHS

Representative config:

```text
polynomial_rls_b128_s1_r0p99_n0p001_d3_aw0p5_rawpoly
```

This is decisive on the controlled frequency task:

- `controlled_frequency`: 0.0030 vs MLP 0.1569, delta +0.1538, wins 3/0/0.

It breaks:

- `synthetic_compositional`: 0.6762 vs MLP 0.2758, delta -0.4004.
- `synthetic_frequency`: 1.1630 vs MLP 1.1493, delta -0.0137.

Conclusion: raw polynomial structure is not optional if the benchmark includes
exact finite-degree algebraic identities. Normalized kernels hide this useful
scale information. But raw tensor algebra is not a compositional learner.

### Arc-cosine / NNGP kernel

Best observed non-MLP compositional rows:

- Budget 128: `arccosine_rls_b128_s1_r0p99_n0p05_arc1_ai4` gets 0.4272 vs
  MLP 0.2758, delta -0.1514.
- Budget 256: `arccosine_rls_b256_s1_r0p99_n0p05_arc1_ai6` gets 0.3725 vs
  MLP 0.2758, delta -0.0967.

Conclusion: NNGP structure moves in the right direction on the tanh
compositional oracle, but at the tested budgets it is still clearly behind a
trained MLP.

### Residual MLP plus kernel hybrid

Best observed hybrid compositional row:

- `hybrid_mlp_h128_algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai8`
  gets 0.3050 vs MLP 0.2758, delta -0.0292.

It also damages other wins:

- `hybrid_mlp_h128_algebraic_green...` on `controlled_frequency`: 0.4038 vs
  MLP 0.1569, delta -0.2470.
- `hybrid_mlp_h128_algebraic_green...` on stateful digit test accuracy loses
  to MLP on label drift, mask noise, and permuted pixels in the hybrid probe.

Conclusion: the residual hybrid is not the path. It is closer on
`synthetic_compositional`, but it pays for that by adding MLP compute and
weakening the non-MLP retention/algebraic advantages. It also does not answer
the original criticism because the best row contains an MLP.

## Exact Configuration Conflicts

The evidence points to four real conflicts:

1. Raw scale versus normalized geometry.
   - Raw polynomial wins `controlled_frequency` by a huge margin.
   - Normalized algebraic-Green wins the broad controlled suite and digits.
   - A single kernel normalization choice does not cover both.

2. Immediate allocation versus temporal reserve.
   - No-throttle algebraic-Green wins IID/class-blocked held-out retention but
     loses label drift, mask noise, and pixel permutation.
   - Throttled algebraic-Green fixes stateful digits.
   - Controlled recursive tasks often benefit from filling the dictionary
     immediately.

3. Local/RKHS memory versus learned composition.
   - Algebraic-Green and raw polynomial are strong on algebraic recursive
     structure.
   - Arc-cosine kernels help but still do not match the trained MLP on the
     two-hidden-layer tanh oracle.
   - Static kernels appear to need either much better compositional features or
     learned feature construction.

4. Retention metric versus current-block metric.
   - `digits_class_blocked` held-out test accuracy strongly favors the kernel.
   - `digits_class_blocked` final-window accuracy and MSE strongly favor the
     MLP because the final window is a current-block specialization metric.
   - A final claim must state which behavioral criterion is intended. For
     continual learning, held-out all-class retention is the more relevant
     criterion, but the final-window loss cannot be ignored if the benchmark bar
     demands every reported metric.

## Remaining Blockers

### Hard blocker: `synthetic_compositional`

Best D07 candidate result, disqualified for a non-MLP claim because it contains
an MLP residual:

```text
0.3050 hybrid_mlp_h128_algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai8
```

Best fair MLP:

```text
0.2758 mlp_h128
```

Gap:

```text
-0.0292 final-window MSE, losses 0/3/0
```

The best pure kernel result is worse:

```text
0.3725 arccosine_rls_b256_s1_r0p99_n0p05_arc1_ai6
```

So the real Step 2 open problem is not algebraic memory or stateful retention.
It is learned compositional feature construction under bounded online compute.

### Soft blocker: `synthetic_frequency`

Best D07 non-MLP result:

```text
1.1590 polynomial_rls_b128_s1_r0p99_n0p05_d3_aw0p5_rawpoly_ai4
```

Best fair MLP:

```text
1.1493 mlp_h64_64
```

Gap:

```text
-0.0097 final-window MSE, seed split 2/1/0
```

This is close enough that a multi-bank learner could plausibly clear it with a
better spectral or random Fourier component. It is not the same kind of hard
blocker as `synthetic_compositional`.

### Metric blocker: class-blocked final-window digits

If the promotion bar is "beat MLP on every metric emitted by the script", D07
does not clear it because `digits_class_blocked` final-window metrics favor MLP:

- Final-window MSE: kernel 0.0583 vs MLP 0.0030, delta -0.0553.
- Final-window accuracy: kernel 0.6456 vs MLP 0.9922, delta -0.3467.

If the bar is "beat MLP on retained held-out performance after class-blocked
continual exposure", D07 clears it decisively:

- Test MSE: kernel 0.0438 vs MLP 0.1343, delta +0.0905.
- Test accuracy: kernel 0.8454 vs MLP 0.1528, delta +0.6926.

This needs to be resolved explicitly in the canonical benchmark contract.

## What It Takes To Beat Best MLP On All Benchmarks

The next canonical learner must not pick a separate method per benchmark. It
needs one fixed update rule and one fixed prediction rule that contains the
winning mechanisms as internal resources. The minimal evidence-backed design is
a single additive, non-router multi-bank learner:

```text
y_hat(x) = y_poly_raw(x) + y_alg_green(x) + y_stateful_green(x) + y_comp(x)
```

Every bank is updated every step from the same scalar error. The resource
manager may allocate centers, bandwidth, forgetting, and budget across banks,
but it should not route predictions by benchmark identity and should not use a
selector over complete MLP experts.

Required banks:

1. Raw polynomial bank.
   - Degree 3, unnormalized, RLS coefficient updates.
   - Purpose: preserve the `controlled_frequency` win.
   - Evidence: 0.0030 vs MLP 0.1569.

2. Normalized algebraic-Green bank.
   - Degree 3 direct-sum polynomial plus algebraic-Green geometry.
   - Purpose: broad controlled recursive suite, synthetic polynomial, IID
     digits, and class-blocked held-out retention.
   - Evidence: positive deltas on controlled nonlinear, interaction, triple,
     rare, polynomial, synthetic polynomial, digits IID, and class-blocked
     held-out accuracy.

3. Temporally reserved stateful bank.
   - Same algebraic-Green family, but with center-add intervals such as 4 or 8
     and forgetting `rho` around 0.99 to 0.995.
   - Purpose: label drift, pixel permutation, and mask noise.
   - Evidence: held-out digit accuracy deltas from +0.0421 to +0.0588.

4. Compositional operator bank.
   - This is the missing piece. D07's arc-cosine and residual hybrids show
     direction but not success.
   - The next trials should target explicit learned or adaptive composition:
     random tanh features with RLS, Chebyshev/KAN-style spline features,
     Volterra or tensor-train features with budgeted rank, spectral mixtures
     for synthetic frequency, and online Nyström features whose centers are
     allocated by utility rather than ALD alone.

Required resource manager:

- Allocate center insertions and per-bank budget by loss reduction per unit
  compute, not by benchmark labels.
- Maintain a reserve so late nonstationary phases get capacity.
- Allow the raw polynomial bank to fill immediately, because throttling is not
  needed for exact algebraic identity tasks.
- Penalize banks that fail to reduce one-step residuals, but keep a floor for
  banks that protect delayed retention. Otherwise the manager will discard the
  stateful bank before it becomes useful.

Required promotion test:

- Run one fixed configuration across the full D07 benchmark suite.
- Compare against the best MLP baseline per benchmark.
- Require positive mean delta and paired seed non-losses on:
  `controlled_*`, `synthetic_polynomial`, `synthetic_frequency`,
  `synthetic_compositional`, `digits_iid`, `digits_label_drift`,
  `digits_permuted_pixels`, and `digits_mask_noise`.
- For `digits_class_blocked`, decide before promotion whether the criterion is
  held-out all-class retention, current-block final-window specialization, or
  both. If both, the learner needs a dual-horizon objective: one fast local
  component for current block performance and one slow retained component for
  all-class memory.

## Recommended D08 Experiments

1. Implement the additive multi-bank learner without MLP experts.
   - Banks: raw polynomial, normalized algebraic-Green, throttled
     algebraic-Green, compositional operator.
   - Prediction is additive; no benchmark router.

2. Attack `synthetic_compositional` directly.
   - Start with random tanh features plus RLS because the target oracle is a
     two-hidden-layer tanh function.
   - Then test Chebyshev/spline and low-rank Volterra banks.
   - The numerical target is strict: beat final-window MSE 0.2758 with a
     non-MLP bank, not merely approach 0.3050.

3. Add a spectral component for `synthetic_frequency`.
   - The current miss is small: 1.1590 vs 1.1493.
   - Random Fourier or learned sinusoidal features should be enough if they do
     not interfere with the algebraic banks.

4. Test resource allocation as a center/budget manager, not a prediction router.
   - Score bank center proposals by residual improvement, novelty, and compute.
   - Keep protected budget floors for raw polynomial and stateful memory.
   - Report active centers, additions, throttled proposals, replacements,
     runtime, and final accuracy/MSE.

5. Run a fixed canonical configuration.
   - The current evidence is promising but still cherry-picks configs.
   - The next result should be one config, one script invocation, all datasets.

## Bottom Line

D07 found real non-MLP structure. Algebraic-Green plus KRLS is not a toy; it
beats fair MLP on many controlled recursive and held-out digit retention
benchmarks. Raw polynomial KRLS solves a task the normalized kernel misses.
Temporal center management fixes stateful external digits. These are genuine
mechanisms.

But D07 is not yet a universal Step 2 solution. The exact missing capability is
bounded online compositional feature construction. A winning D08 needs to
combine the proven banks into one additive learner and add a compositional
operator bank that beats MLP on `synthetic_compositional` while preserving the
raw polynomial and throttled-memory wins.

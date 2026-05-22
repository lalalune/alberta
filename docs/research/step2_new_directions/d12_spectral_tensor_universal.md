# D12 Spectral/Tensor Universal Learner Assessment

## Verdict

D12 is **not** a canonical Step 2 solution. It is a useful non-MLP, non-router
learner family that wins several frequency/polynomial regimes, but it does not
beat the fair MLP on the decisive blocker, `synthetic_compositional`, and it
also loses the stateful external `digits_label_drift` check.

The best role for D12 is as a component bank for a larger learned resource
manager: the `fourier_wavelet` NLMS configuration is strong and cheap on
frequency/polynomial tasks, but the spectral/tensor mechanism here does not
provide arbitrary recursive feature discovery by itself.

## Implementation

File: `examples/The Alberta Plan/Step2/new_directions/d12_spectral_tensor_universal.py`

The implemented learner is one online predictor:

- It builds a deterministic candidate feature pool from compact basis families.
- It maintains one active budgeted subset.
- It updates candidate utility traces, active feature selection, and readout
  weights under the same prequential loss.
- It supports RLS, normalized LMS, and compact Autostep-style per-feature
  adaptation.
- It does not use MLP predictions, residual MLPs, expert routing, offline labels,
  or per-dataset switches.

New basis families added in this pass:

- `chaos`: sparse Chebyshev polynomial-chaos interaction products.
- `ridgelet`: random ridgelet/Hermite spectral projections.
- `tensor_cp`: compressed CP-style products of nonlinear ridge factors.
- `deep_ridgelet`: fixed two-level compositional ridgelets with a linear online
  readout.
- `ridgelet_chaos`: combined chaos, ridgelet, tensor-CP, wavelet, and deep
  ridgelet features.
- `spectral_tensor`: enriched to include the new compact basis blocks in
  addition to the existing Fourier, RFF, ANOVA, TensorSketch, Chebyshev, and
  wavelet blocks.

New CLI knobs:

- `--chaos-count`
- `--ridgelet-count`
- `--ridgelet-scale`
- `--tensor-cp-count`
- `--tensor-cp-scale`
- `--deep-ridgelet-count`
- `--deep-ridgelet-inner`
- `--deep-ridgelet-scale`

## Primary Blocker: `synthetic_compositional`

Protocol for the main runs: 3 paired seeds, 1200 online steps, final window 300,
fair MLP grid `mlp_h64`, `mlp_h128`, `mlp_h64_64`.

This is the right stress test for D12 because the stream source is a two-hidden-
layer tanh oracle:

`inner = tanh(V x + c)`

`outer = tanh(W inner + b)`

`target_k = a . outer + noise`

A shallow spectral/polynomial/tensor feature bank can approximate parts of this
class, but the benchmark is explicitly designed so shallow feature classes are
not sufficient.

### Compositional Ablation Results

| Run | Best D12 Method | Best D12 Final MSE | Best MLP Final MSE | Paired Diff | Wins/Losses/Ties | Result JSON |
|---|---|---:|---:|---:|---:|---|
| Original RLS basis sweep | `spec_rff_tensor_rls_b256_c768_d5_o3_rho0p995_eta0p2_s0` | 0.4746 +/- 0.1406 | 0.2758 +/- 0.0936 | -0.1845 +/- 0.0505 | 0/3/0 | `outputs/step2_new_directions/d12_compositional_basis_3seed/results.json` |
| New chaos/ridgelet/tensor bases | `spec_spectral_tensor_nlms_b512_c1536_d5_o4_rho0p995_eta0p5_s0` | 0.4482 +/- 0.1291 | 0.2758 +/- 0.0936 | -0.1765 +/- 0.0339 | 0/3/0 | `outputs/step2_new_directions/d12_new_basis_comp_3seed/results.json` |
| No-replacement active set | `spec_fourier_wavelet_nlms_b512_c1536_d5_o4_rho0p995_eta0p5_s0` | 0.4257 +/- 0.1274 | 0.2758 +/- 0.0936 | -0.1540 +/- 0.0322 | 0/3/0 | `outputs/step2_new_directions/d12_no_replacement_comp_3seed/results.json` |
| Deep ridgelet, eta 1.0, no replacement | `spec_spectral_tensor_nlms_b512_c2048_d5_o4_rho0p995_eta1_s0` | 0.6139 +/- 0.2357 | 0.2758 +/- 0.0936 | -0.3165 +/- 0.1158 | 0/3/0 | `outputs/step2_new_directions/d12_deep_ridgelet_comp_eta1_3seed/results.json` |
| Deep ridgelet, eta 0.5, replacement | `spec_spectral_tensor_nlms_b512_c2048_d5_o4_rho0p995_eta0p5_s0` | 0.5316 +/- 0.1758 | 0.2758 +/- 0.0936 | -0.2599 +/- 0.0818 | 0/3/0 | `outputs/step2_new_directions/d12_deep_ridgelet_comp_eta05_replace_3seed/results.json` |
| Deep ridgelet RLS | `spec_deep_ridgelet_rls_b256_c2048_d5_o4_rho0p995_eta0p2_s0` | 0.6303 +/- 0.2014 | 0.2758 +/- 0.0936 | -0.3586 +/- 0.1071 | 0/3/0 | `outputs/step2_new_directions/d12_deep_ridgelet_rls_comp_3seed/results.json` |
| Large active budget | `spec_rff_tensor_nlms_b1536_c2048_d5_o4_rho0p995_eta0p5_s0` | 0.4613 +/- 0.1349 | 0.2758 +/- 0.0936 | -0.1896 +/- 0.0408 | 0/3/0 | `outputs/step2_new_directions/d12_large_budget_comp_3seed/results.json` |

Interpretation:

- The best D12 compositional run is the no-replacement `fourier_wavelet` NLMS
  run at 0.4257 final-window MSE, still far behind the best fair MLP at 0.2758.
- Expanding the budget to 1536 active features did not help.
- RLS did not rescue the random deep ridgelet basis.
- The explicit two-level random ridgelet basis did not outperform the broader
  shallow spectral/tensor mix, which suggests the issue is not just the absence
  of a two-level functional form. The missing ingredient is learned, task-
  specific recursive feature construction and retention, not more fixed random
  features.

## Follow-Up Blockers Where D12 Works

Run: `outputs/step2_new_directions/d12_followup_blockers_3seed/results.json`

Protocol: 3 seeds, 1200 steps, final window 300, no active replacement after
fill, 512 active features, 1536 candidates, NLMS step size 0.5.

| Dataset | Best D12 Method | Best D12 Final MSE | Best MLP Final MSE | Paired Diff | Wins/Losses/Ties |
|---|---|---:|---:|---:|---:|
| `controlled_frequency` | `spec_fourier_wavelet_nlms_b512_c1536_d5_o4_rho0p995_eta0p5_s0` | 0.0237 +/- 0.0015 | 0.1569 +/- 0.0264 | +0.1332 +/- 0.0277 | 3/0/0 |
| `controlled_polynomial` | `spec_fourier_wavelet_nlms_b512_c1536_d5_o4_rho0p995_eta0p5_s0` | 0.6676 +/- 0.1653 | 0.8611 +/- 0.1893 | +0.1923 +/- 0.0342 | 3/0/0 |
| `synthetic_frequency` | best by seed split across `cheb_anova` and `fourier_wavelet`; lowest mean was `spectral_tensor` | 0.8618 +/- 0.2000 | 1.1493 +/- 0.2579 | +0.2968 +/- 0.0794 | 3/0/0 |

Interpretation:

- The spectral/tensor idea is real on tasks whose target lies near Fourier,
  polynomial, or low-order interaction bases.
- `fourier_wavelet` is the most attractive D12 component: it is simple, cheap,
  and won both controlled frequency and controlled polynomial.
- These wins do not solve Step 2 because the same mechanism fails the
  compositional blocker and external stateful digits.

## Stateful External Check

Run: `outputs/step2_new_directions/d12_digits_label_drift_3seed/results.json`

Protocol: 3 seeds, 1200 steps, final window 300, stateful sklearn digits label
drift, NLMS step size 0.5, 512 active features.

| Metric | Best D12 | Best MLP | Paired Diff | Wins/Losses/Ties |
|---|---:|---:|---:|---:|
| Final-window MSE | 0.0555 +/- 0.0004 (`fourier_wavelet`) | 0.0383 +/- 0.0010 (`mlp_h64`) | -0.0174 +/- 0.0006 | 0/3/0 |
| Final-window accuracy | 0.7367 +/- 0.0019 (`fourier_wavelet`) | 0.8600 +/- 0.0117 (`mlp_h64`) | negative | 0/3/0 |
| Test accuracy | 0.8281 +/- 0.0243 (`fourier_wavelet`) | 0.9041 +/- 0.0043 (`mlp_h128`) | -0.0711 +/- 0.0215 | 0/3/0 |

Interpretation:

- D12 does not preserve the MLP's external classifier performance.
- The high-dimensional digits stream exposes a weakness of compact fixed bases:
  the learner can track some label signal but lacks the learned representation
  quality of the MLP.

## Runtime And Compute

Wall-clock times for the major runs:

| Run | Wall Clock |
|---|---:|
| New compositional basis sweep | 48.30 s |
| No-replacement compositional sweep | 56.04 s |
| Deep ridgelet eta 1.0 compositional sweep | 51.19 s |
| Deep ridgelet eta 0.5 replacement sweep | 56.18 s |
| Deep ridgelet RLS compositional sweep | 65.31 s |
| Large-budget compositional sweep | 45.70 s |
| Follow-up frequency/polynomial blockers | 96.99 s |
| Digits label drift | 45.24 s |

Representative per-method runtimes:

- On `controlled_frequency`, `fourier_wavelet` took 0.4467 +/- 0.0972 s per
  seed and beat the best MLP. The best MLP (`mlp_h64_64`) took 0.6007 +/- 0.3336
  s per seed.
- On `controlled_polynomial`, `fourier_wavelet` took 0.4473 +/- 0.1229 s per
  seed and beat the best MLP. The best MLP (`mlp_h64`) took 0.3747 +/- 0.0907 s
  per seed.
- On `synthetic_compositional`, the stronger `spectral_tensor` configurations
  took roughly 4 to 5 s per seed and still lost to MLP.
- On `digits_label_drift`, `fourier_wavelet` was faster than MLP but much less
  accurate.

Compute conclusion:

- D12 can be computationally attractive when the basis matches the task.
- The expensive variants do not buy enough accuracy on the hard blocker.
- Increasing active budget worsens the compute/accuracy tradeoff on
  compositional targets.

## What Is Missing

D12 is missing the part Step 2 needs most: learned recursive feature discovery.

The current D12 learner can select among a large fixed basis, but it does not
learn inner features that become parents of later features in a task-specific
way. The `deep_ridgelet` basis tested this hypothesis with fixed random two-
level functions. It still failed, which means a static random approximation of
the right functional family is not enough for the continual setting.

Concrete gaps:

- No learned inner representation: the basis can include nested nonlinear
  functions, but their inner projections are fixed at birth.
- No future-utility credit for recursive parents: selection is driven by current
  residual correlation and active coefficient utility, not by whether a feature
  enables later useful features.
- No high-dimensional representation learning: digits label drift loses held-
  out accuracy by about 7.1 percentage points against the best MLP.
- No canonical single configuration: the useful basis changes by regime, and
  no fixed D12 configuration beats best MLP across compositional, frequency,
  polynomial, and digits.
- No robust retained memory advantage: no-replacement helps compositional
  slightly, but not enough; replacement hurts some runs, but retaining a fixed
  basis also fails.

## Recommendation

D12 should not be promoted as the Alberta Plan Step 2 solution.

The part worth keeping is the `fourier_wavelet` NLMS bank:

- It is cheap.
- It cleanly wins controlled frequency, controlled polynomial, and synthetic
  frequency.
- It is a good component for a multi-bank learner.

The canonical path should combine D12's `fourier_wavelet` bank with a mechanism
that actually learns recursive features:

- A learned resource manager should allocate budget to `fourier_wavelet` when
  spectral structure is detected.
- A separate recursive feature-birth or learned-composition bank is still needed
  for `synthetic_compositional`.
- The external digits bank likely needs either learned representation features
  or the D07-style kernel/dictionary mechanism that previously performed better
  on stateful digits.

Final assessment: D12 is mathematically different and useful, but it does not
close Step 2 beyond a shadow of a doubt. It should feed the multi-bank design
rather than become the canonical learner.

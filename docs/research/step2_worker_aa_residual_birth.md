# Step 2 Worker AA Residual-Birth Learner

Status: **negative to partial** unless a follow-up expansion overturns the 3-seed blocker matrix.

## Protocol

Primary matrix command executed from the repository root:

```bash
source .venv/bin/activate && python output/subagents/step2_worker_aa_residual_birth.py --datasets digits_class_blocked,digits_mask_noise,synthetic_polynomial,synthetic_frequency,synthetic_compositional,controlled_rare --steps 1200 --n-seeds 3 --final-window 300 --output-dir outputs/step2_worker_aa_residual_birth_3seed --note-path docs/research/step2_worker_aa_residual_birth.md
```

Datasets: digits_class_blocked, digits_mask_noise, synthetic_polynomial, synthetic_frequency, synthetic_compositional, controlled_rare. Seeds: 0..2; steps=1200; final_window=300.

Fair MLP comparator: per-seed best of `mlp_h64`, `mlp_h128`, and `mlp_h64_64`, using the same materialized stream.

Candidate: one additive prediction path over a bias plus residual-born local features.  Birth occurs only when the prequential residual is high relative to its running scale and the current representation is novel. A born feature is centered at the current clipped observation plus earlier feature activations, seeds its output weights from the residual, and then uses the same shared readout update as every other feature.  Mature low-utility features are replaced when the budget is full.

Fixed candidate knobs: `budget=192`, `readout_update=rls`, `rls_rho=0.995`, `rls_delta=20`, `sigma=2.0`, `birth_residual_z=-0.15`, `novelty_threshold=0.01`, `birth_interval=1`, `birth_gain=0.35`, `utility_decay=0.995`.

Positive paired deltas below favor residual birth over the best fair MLP. For MSE, the delta is best MLP minus residual birth; for accuracy, it is residual birth minus best MLP.

## Results

### controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Features | Births | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  |  | 0.2615 +/- 0.0235 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  |  | 0.3057 +/- 0.0807 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  |  | 0.3088 +/- 0.0345 |
| `residual_birth_recursive` | 0.0659 +/- 0.0031 | 0.1235 +/- 0.0343 |  |  | 60.3333 +/- 0.6667 | 60.3333 +/- 0.6667 | 1.0619 +/- 0.0372 |

`final_window_mse`: +0.0073 +/- 0.0135; 2/1/0.

### digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Features | Births | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  |  | 0.8724 +/- 0.4257 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  |  | 0.7828 +/- 0.3511 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  |  | 0.5062 +/- 0.0081 |
| `residual_birth_recursive` | 0.0016 +/- 0.0002 | 0.0012 +/- 0.0001 | 0.9900 +/- 0.0000 | 0.1002 +/- 0.0000 | 192.0000 +/- 0.0000 | 583.3333 +/- 11.7237 | 3.7670 +/- 0.1588 |

`final_window_mse`: +0.0014 +/- 0.0001; 3/0/0.
`test_accuracy`: -0.0656 +/- 0.0208; 0/3/0.

### digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Features | Births | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  |  | 0.3665 +/- 0.0327 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  |  | 0.4708 +/- 0.0141 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  |  | 0.4338 +/- 0.0521 |
| `residual_birth_recursive` | 0.0965 +/- 0.0040 | 0.0830 +/- 0.0017 | 0.4678 +/- 0.0318 | 0.4156 +/- 0.1484 | 192.0000 +/- 0.0000 | 344.3333 +/- 23.7861 | 3.4869 +/- 0.3030 |

`final_window_mse`: -0.0487 +/- 0.0032; 0/3/0.
`test_accuracy`: -0.4045 +/- 0.1473; 0/3/0.

### synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Features | Births | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  |  | 0.4168 +/- 0.1865 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  |  | 0.4868 +/- 0.2388 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  |  | 0.3493 +/- 0.0578 |
| `residual_birth_recursive` | 1.0318 +/- 0.4565 | 0.5071 +/- 0.1267 |  |  | 99.3333 +/- 1.4530 | 99.3333 +/- 1.4530 | 1.7748 +/- 0.1230 |

`final_window_mse`: -0.7601 +/- 0.3625; 0/3/0.

### synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Features | Births | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  |  | 0.3095 +/- 0.1329 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  |  | 0.3164 +/- 0.1548 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  |  | 0.3107 +/- 0.0286 |
| `residual_birth_recursive` | 1.3951 +/- 0.0700 | 1.3860 +/- 0.2549 |  |  | 91.3333 +/- 0.8819 | 91.3333 +/- 0.8819 | 1.3796 +/- 0.1065 |

`final_window_mse`: -0.2467 +/- 0.2493; 1/2/0.

### synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Features | Births | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  |  | 0.3893 +/- 0.2103 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  |  | 0.4487 +/- 0.2105 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  |  | 0.2884 +/- 0.0089 |
| `residual_birth_recursive` | 1.2694 +/- 0.9523 | 1.0495 +/- 0.2417 |  |  | 141.0000 +/- 1.0000 | 141.0000 +/- 1.0000 | 1.8682 +/- 0.1153 |

`final_window_mse`: -0.3220 +/- 0.3079; 1/2/0.

## Assessment

The fixed residual-birth learner beat the best fair MLP on 2/6 datasets by mean final-window MSE.

I did not expand beyond 3 seeds because the result is not promising at the universal Step 2 bar: it wins `digits_class_blocked` and narrowly wins `controlled_rare`, but loses all three synthetic structure regimes and collapses on `digits_mask_noise` retained accuracy.

This is genuinely less hand-engineered than D18 in mechanism count: it has no fixed polynomial/Fourier/tanh bank, no resource manager, no learned block gains, no simplex projection, no trace-specific output add-on, and no per-dataset routing.  The price is that the mechanism is much less structured: all discoveries are local bumps in the current representation, so it can memorize or patch residual regions but does not reliably infer compact algebraic or periodic structure.

The recursive principle is plausible but weak in this implementation. A new feature can depend on earlier features because its center includes their current activations, and future residuals can therefore trigger features over features.  However, utility is still local prediction utility rather than a deeper estimate of reusable explanatory value. That makes the rule closer to residual dictionary growth than a full recursive feature-discovery theory.

## Next Best Changes

1. Replace scalar RBF bandwidth with an online adaptive bandwidth derived from accepted-center distances, still using one birth rule.
2. Add an IDBD/Autostep-style per-feature step-size to separate useful features from stale high-activation features without adding expert families.
3. Make utility predict future residual reduction over a short trace, not instantaneous residual activation, before increasing the budget or adding any new basis family.

## Artifacts

- Results JSON: `outputs/step2_worker_aa_residual_birth_3seed/results.json`
- Records JSONL: `outputs/step2_worker_aa_residual_birth_3seed/records.jsonl`
- Runner: `/Users/shawwalters/Desktop/nca_fun/alberta-framework/output/subagents/step2_worker_aa_residual_birth.py`

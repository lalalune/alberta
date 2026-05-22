# Worker Z: Fixed Random-Feature NLMS

Scope: random-feature/reservoir single learner experiments only. This run uses one stateless fixed random feature expansion, one linear multi-head readout, and one normalized LMS update everywhere.

## Files

- Runner: `output/subagents/step2_worker_z_random_features.py`
- Results: `outputs/step2_worker_z_random_features_wide_nopoly_3seed`
- JSON: `outputs/step2_worker_z_random_features_wide_nopoly_3seed/results.json`
- Summary: `outputs/step2_worker_z_random_features_wide_nopoly_3seed/SUMMARY.md`
- Report: `docs/research/step2_worker_z_random_features.md`

## Command

```bash
source .venv/bin/activate && python output/subagents/step2_worker_z_random_features.py --datasets digits_class_blocked,digits_mask_noise,synthetic_polynomial,synthetic_frequency,synthetic_compositional,controlled_rare --steps 1200 --n-seeds 3 --seed 0 --final-window 300 --output-dir outputs/step2_worker_z_random_features_wide_nopoly_3seed --note-path docs/research/step2_worker_z_random_features.md --tanh-width 1536 --relu-width 512 --fourier-width 1536 --poly-width 0 --step-size 0.8 --weight-decay 1.0
```

## Protocol

- Datasets: digits_class_blocked, digits_mask_noise, synthetic_polynomial, synthetic_frequency, synthetic_compositional, controlled_rare; 3 paired seeds; 1200 online steps; final window 300.
- Fair MLPs: `mlp_h64`, `mlp_h128`, `mlp_h64_64` from the D07/D18 harness (`MultiHeadMLPLearner`, LayerNorm, ObGD, step size 0.03, sparsity 0.5).
- Random map: bias + raw clipped input + tanh random features + ReLU random features + sine/cosine Fourier random features.
- Widths: tanh=1536, ReLU=512, Fourier=1536 (sine and cosine both included), polynomial-ridge=0 (squared and cubed projections); update=NLMS with eta=0.8, weight_decay=1.0.

Positive paired differences favor `random_features_nlms`; for MSE this is best fair MLP minus candidate.

## Results

### controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.1219 +/- 0.0063 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.1860 +/- 0.0496 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.1731 +/- 0.0148 |
| `random_features_nlms` | 0.0577 +/- 0.0161 | 0.0832 +/- 0.0232 |  |  |  | 0.1467 +/- 0.0032 |

`final_window_mse` random-feature-vs-best-MLP diff: +0.0154 +/- 0.0082; wins/losses/ties 2/1/0; best MLP counts {'mlp_h64': 3}.

Verdict: beats/ties fair MLP on paired final-window MSE.

### digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 | 0.1362 +/- 0.0042 | 0.7765 +/- 0.3990 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 | 0.1343 +/- 0.0019 | 0.6511 +/- 0.2183 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 | 0.1657 +/- 0.0019 | 0.4752 +/- 0.0248 |
| `random_features_nlms` | 0.0031 +/- 0.0000 | 0.0031 +/- 0.0000 | 0.9900 +/- 0.0019 | 0.1323 +/- 0.0166 | 0.1266 +/- 0.0003 | 0.1961 +/- 0.0056 |

`final_window_mse` random-feature-vs-best-MLP diff: -0.0001 +/- 0.0001; wins/losses/ties 2/1/0; best MLP counts {'mlp_h64_64': 3}.
`test_accuracy` random-feature-vs-best-MLP diff: -0.0334 +/- 0.0075; wins/losses/ties 0/3/0.

Verdict: does not beat fair MLP on paired final-window MSE.

### digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 | 0.0448 +/- 0.0029 | 0.4485 +/- 0.0384 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 | 0.0447 +/- 0.0012 | 0.4223 +/- 0.0393 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 | 0.0481 +/- 0.0028 | 0.5543 +/- 0.0355 |
| `random_features_nlms` | 0.0448 +/- 0.0008 | 0.0528 +/- 0.0015 | 0.8189 +/- 0.0116 | 0.8547 +/- 0.0174 | 0.0408 +/- 0.0023 | 0.2060 +/- 0.0198 |

`final_window_mse` random-feature-vs-best-MLP diff: +0.0030 +/- 0.0005; wins/losses/ties 3/0/0; best MLP counts {'mlp_h128': 3}.
`test_accuracy` random-feature-vs-best-MLP diff: +0.0346 +/- 0.0045; wins/losses/ties 3/0/0.

Verdict: beats/ties fair MLP on paired final-window MSE.

### synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.2772 +/- 0.1263 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.2664 +/- 0.1292 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.2312 +/- 0.0079 |
| `random_features_nlms` | 0.2332 +/- 0.0735 | 0.1485 +/- 0.0253 |  |  |  | 0.1586 +/- 0.0034 |

`final_window_mse` random-feature-vs-best-MLP diff: +0.0385 +/- 0.0224; wins/losses/ties 3/0/0; best MLP counts {'mlp_h128': 1, 'mlp_h64': 2}.

Verdict: beats/ties fair MLP on paired final-window MSE.

### synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.2409 +/- 0.1151 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.2279 +/- 0.0965 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.1705 +/- 0.0056 |
| `random_features_nlms` | 1.4731 +/- 0.3501 | 1.6900 +/- 0.3479 |  |  |  | 0.1440 +/- 0.0023 |

`final_window_mse` random-feature-vs-best-MLP diff: -0.3246 +/- 0.0922; wins/losses/ties 0/3/0; best MLP counts {'mlp_h64': 1, 'mlp_h64_64': 2}.

Verdict: does not beat fair MLP on paired final-window MSE.

### synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.4044 +/- 0.2060 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.3748 +/- 0.1667 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.3527 +/- 0.0697 |
| `random_features_nlms` | 1.1937 +/- 0.7731 | 1.2461 +/- 0.3107 |  |  |  | 0.1518 +/- 0.0005 |

`final_window_mse` random-feature-vs-best-MLP diff: -0.2462 +/- 0.1290; wins/losses/ties 0/3/0; best MLP counts {'mlp_h64_64': 3}.

Verdict: does not beat fair MLP on paired final-window MSE.

## Suite Verdict

Rows with positive paired final-window MSE diff: 3/6. Rows with negative diff: 3/6.

This fixed random-feature NLMS learner is mathematically simple and universal in the approximation-theory sense: non-polynomial random ridge functions and Fourier features span dense approximating families for broad continuous nonlinear targets as width grows, while the linear readout keeps online adaptation convex. The NLMS normalizer makes the step invariant to feature-vector scale, so the same update can be used on low-dimensional synthetic streams and standardized digit pixels.

The continual-learning story is retention/reactivity rather than feature birth: fixed random features retain old functions in the readout weights, and constant-step NLMS can overwrite predictions when targets or masks change. The cost is that finite random bases may need many features to express sharp algebraic structure, and the single global step size has no route-specific rescue mechanism.

Conclusion: it beats fair MLP only if every blocker row is nonnegative against the per-seed best fair MLP. See the suite verdict above for the actual result.

## Additional Global Checks

I also ran two additional fixed global variants, both still single random-feature NLMS learners with no router or per-dataset selection. The table includes the current wide run for comparison.

```bash
source .venv/bin/activate && python output/subagents/step2_worker_z_random_features.py --datasets digits_class_blocked,digits_mask_noise,synthetic_polynomial,synthetic_frequency,synthetic_compositional,controlled_rare --steps 1200 --n-seeds 3 --seed 0 --final-window 300 --output-dir outputs/step2_worker_z_random_features_nopoly_3seed --note-path docs/research/step2_worker_z_random_features.md --poly-width 0
source .venv/bin/activate && python output/subagents/step2_worker_z_random_features.py --datasets digits_class_blocked,digits_mask_noise,synthetic_polynomial,synthetic_frequency,synthetic_compositional,controlled_rare --steps 1200 --n-seeds 3 --seed 0 --final-window 300 --output-dir outputs/step2_worker_z_random_features_poly_3seed --note-path docs/research/step2_worker_z_random_features.md
```

| Variant | Output | Positive / negative FW-MSE rows | Main losses |
|---|---|---:|---|
| default no-poly map | `outputs/step2_worker_z_random_features_nopoly_3seed` | 3 / 3 | `digits_class_blocked` (-0.0001), `synthetic_frequency` (-0.3559), `synthetic_polynomial` (-0.2886) |
| polynomial-ridge map | `outputs/step2_worker_z_random_features_poly_3seed` | 2 / 4 | `digits_class_blocked` (-0.0015), `synthetic_compositional` (-0.0952), `synthetic_frequency` (-0.8126), `synthetic_polynomial` (-0.1903) |
| wide no-poly map | `outputs/step2_worker_z_random_features_wide_nopoly_3seed` | 3 / 3 | `digits_class_blocked` (-0.0001), `synthetic_frequency` (-0.3246), `synthetic_polynomial` (-0.2462) |

The wide no-poly map is the least bad variant, but it is not promising enough for a 10-seed risk run because it still loses all three paired seeds on both synthetic blocker rows and remains negative on the class-blocked final-window comparison.

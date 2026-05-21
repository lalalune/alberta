# D07 Budgeted Diffusion-KRLS Results

Protocol: 10 paired seeds, 2500 online steps, final window 500. Kernel=algebraic_green, updates=rls, budgets=[128], sigmas=[0.5], rhos=[1.0], novelty thresholds=[0.001].

This is a single-learner test. The kernel methods do not consume MLP predictions, routes, stacker weights, or offline labels. Positive kernel-vs-MLP paired differences favor the kernel method.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0985 +/- 0.0081 | 0.3503 +/- 0.0091 |  |  |  | 0.4190 +/- 0.0702 |
| `mlp_h128` | 0.1062 +/- 0.0090 | 0.3716 +/- 0.0112 |  |  |  | 0.4664 +/- 0.0661 |
| `mlp_h64_64` | 0.0796 +/- 0.0046 | 0.2482 +/- 0.0130 |  |  |  | 0.6045 +/- 0.0988 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.2440 +/- 0.0059 | 0.3351 +/- 0.0047 |  |  | 128.0000 +/- 0.0000 | 2.3369 +/- 0.1433 |

`final_window_mse` best-kernel-vs-best-MLP diff: -0.1715 +/- 0.0062; wins/losses/ties 0/10/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2539 +/- 0.0207 | 0.4796 +/- 0.0114 |  |  |  | 0.2615 +/- 0.0159 |
| `mlp_h128` | 0.3902 +/- 0.0210 | 0.5591 +/- 0.0139 |  |  |  | 0.2679 +/- 0.0244 |
| `mlp_h64_64` | 0.5623 +/- 0.0242 | 0.6698 +/- 0.0206 |  |  |  | 0.4455 +/- 0.0563 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.2270 +/- 0.0166 | 0.2802 +/- 0.0076 |  |  | 128.0000 +/- 0.0000 | 1.7285 +/- 0.2016 |

`final_window_mse` best-kernel-vs-best-MLP diff: +0.0269 +/- 0.0076; wins/losses/ties 9/1/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 10}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0417 +/- 0.0016 | 0.0813 +/- 0.0018 |  |  |  | 0.3540 +/- 0.1435 |
| `mlp_h128` | 0.0603 +/- 0.0022 | 0.0957 +/- 0.0017 |  |  |  | 0.3863 +/- 0.1275 |
| `mlp_h64_64` | 0.0728 +/- 0.0050 | 0.1077 +/- 0.0028 |  |  |  | 0.4722 +/- 0.0626 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.0352 +/- 0.0017 | 0.0484 +/- 0.0008 |  |  | 128.0000 +/- 0.0000 | 1.3746 +/- 0.1633 |

`final_window_mse` best-kernel-vs-best-MLP diff: +0.0066 +/- 0.0013; wins/losses/ties 9/1/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 10}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.7019 +/- 0.0931 | 0.8997 +/- 0.0337 |  |  |  | 0.3049 +/- 0.0365 |
| `mlp_h128` | 0.8197 +/- 0.1076 | 0.9950 +/- 0.0379 |  |  |  | 0.3243 +/- 0.0339 |
| `mlp_h64_64` | 1.0018 +/- 0.1144 | 1.0850 +/- 0.0421 |  |  |  | 0.3910 +/- 0.0303 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.4415 +/- 0.0661 | 0.4936 +/- 0.0252 |  |  | 128.0000 +/- 0.0000 | 1.7430 +/- 0.2300 |

`final_window_mse` best-kernel-vs-best-MLP diff: +0.2604 +/- 0.0378; wins/losses/ties 10/0/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 10}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0446 +/- 0.1011 | 1.0901 +/- 0.0377 |  |  |  | 0.3186 +/- 0.0245 |
| `mlp_h128` | 1.0851 +/- 0.1027 | 1.1080 +/- 0.0382 |  |  |  | 0.3626 +/- 0.0388 |
| `mlp_h64_64` | 0.7412 +/- 0.0805 | 0.8529 +/- 0.0364 |  |  |  | 0.5172 +/- 0.0370 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.3125 +/- 0.0440 | 0.4294 +/- 0.0250 |  |  | 128.0000 +/- 0.0000 | 2.6126 +/- 0.3044 |

`final_window_mse` best-kernel-vs-best-MLP diff: +0.4286 +/- 0.0420; wins/losses/ties 10/0/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 10}.

## Interpretation Bar

A positive result here is still a search result unless one fixed kernel configuration beats the best fair MLP across the broad suite. The `best_kernel_vs_best_mlp` rows are useful for detecting whether the mathematical mechanism has headroom; a universal learner claim requires promoting one canonical configuration and rerunning it without per-dataset selection.

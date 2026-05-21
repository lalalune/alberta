# D07 Budgeted Diffusion-KRLS Results

Protocol: 3 paired seeds, 2500 online steps, final window 500. Kernel=algebraic_green, updates=rls, budgets=[128], sigmas=[0.25, 0.5], rhos=[1.0], novelty thresholds=[0.001].

This is a single-learner test. The kernel methods do not consume MLP predictions, routes, stacker weights, or offline labels. Positive kernel-vs-MLP paired differences favor the kernel method.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2636 +/- 0.0189 | 0.5002 +/- 0.0230 |  |  |  | 0.1881 +/- 0.0235 |
| `mlp_h128` | 0.4242 +/- 0.0264 | 0.5844 +/- 0.0243 |  |  |  | 0.2314 +/- 0.0448 |
| `mlp_h64_64` | 0.5713 +/- 0.0381 | 0.6855 +/- 0.0256 |  |  |  | 0.2837 +/- 0.0121 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75` | 0.2503 +/- 0.0208 | 0.3129 +/- 0.0120 |  |  | 128.0000 +/- 0.0000 | 1.2772 +/- 0.2069 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai8` | 0.2994 +/- 0.0175 | 0.4469 +/- 0.0269 |  |  | 128.0000 +/- 0.0000 | 1.0172 +/- 0.1583 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai16` | 0.3260 +/- 0.0161 | 0.5279 +/- 0.0253 |  |  | 128.0000 +/- 0.0000 | 1.1112 +/- 0.1436 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.2362 +/- 0.0208 | 0.3027 +/- 0.0112 |  |  | 128.0000 +/- 0.0000 | 1.0953 +/- 0.1661 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai8` | 0.2899 +/- 0.0172 | 0.4435 +/- 0.0272 |  |  | 128.0000 +/- 0.0000 | 2.0858 +/- 0.2632 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai16` | 0.3167 +/- 0.0154 | 0.5274 +/- 0.0252 |  |  | 128.0000 +/- 0.0000 | 1.0915 +/- 0.0653 |

`final_window_mse` best-kernel-vs-best-MLP diff: +0.0274 +/- 0.0039; wins/losses/ties 3/0/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 3}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0418 +/- 0.0023 | 0.0791 +/- 0.0017 |  |  |  | 0.7046 +/- 0.4435 |
| `mlp_h128` | 0.0617 +/- 0.0038 | 0.0953 +/- 0.0021 |  |  |  | 0.6377 +/- 0.3081 |
| `mlp_h64_64` | 0.0697 +/- 0.0067 | 0.1039 +/- 0.0042 |  |  |  | 0.5050 +/- 0.1490 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75` | 0.0380 +/- 0.0013 | 0.0479 +/- 0.0009 |  |  | 128.0000 +/- 0.0000 | 1.0517 +/- 0.0863 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai8` | 0.0481 +/- 0.0051 | 0.0837 +/- 0.0052 |  |  | 128.0000 +/- 0.0000 | 1.6761 +/- 0.3098 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai16` | 0.0633 +/- 0.0064 | 0.1187 +/- 0.0083 |  |  | 128.0000 +/- 0.0000 | 0.8094 +/- 0.0800 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.0360 +/- 0.0010 | 0.0463 +/- 0.0008 |  |  | 128.0000 +/- 0.0000 | 1.2133 +/- 0.1604 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai8` | 0.0471 +/- 0.0049 | 0.0833 +/- 0.0052 |  |  | 128.0000 +/- 0.0000 | 1.1448 +/- 0.1573 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai16` | 0.0627 +/- 0.0063 | 0.1188 +/- 0.0083 |  |  | 128.0000 +/- 0.0000 | 1.6921 +/- 0.5004 |

`final_window_mse` best-kernel-vs-best-MLP diff: +0.0058 +/- 0.0013; wins/losses/ties 3/0/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 3}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.6132 +/- 0.0707 | 0.9243 +/- 0.0824 |  |  |  | 0.1836 +/- 0.0104 |
| `mlp_h128` | 0.7023 +/- 0.0747 | 1.0035 +/- 0.0874 |  |  |  | 0.2244 +/- 0.0381 |
| `mlp_h64_64` | 0.8931 +/- 0.0930 | 1.0848 +/- 0.0995 |  |  |  | 0.3715 +/- 0.0901 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75` | 0.3966 +/- 0.0285 | 0.5348 +/- 0.0650 |  |  | 128.0000 +/- 0.0000 | 0.9778 +/- 0.1156 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai8` | 0.5068 +/- 0.0571 | 0.7920 +/- 0.0569 |  |  | 128.0000 +/- 0.0000 | 1.1007 +/- 0.1044 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai16` | 0.6459 +/- 0.0501 | 0.9958 +/- 0.0530 |  |  | 128.0000 +/- 0.0000 | 1.1201 +/- 0.3490 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.3770 +/- 0.0251 | 0.5183 +/- 0.0631 |  |  | 128.0000 +/- 0.0000 | 1.4731 +/- 0.1454 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai8` | 0.4955 +/- 0.0568 | 0.7930 +/- 0.0568 |  |  | 128.0000 +/- 0.0000 | 1.2731 +/- 0.0369 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai16` | 0.6463 +/- 0.0492 | 1.0055 +/- 0.0529 |  |  | 128.0000 +/- 0.0000 | 0.9992 +/- 0.1369 |

`final_window_mse` best-kernel-vs-best-MLP diff: +0.2362 +/- 0.0502; wins/losses/ties 3/0/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 3}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1594 +/- 0.1932 | 1.1405 +/- 0.0950 |  |  |  | 0.2254 +/- 0.0388 |
| `mlp_h128` | 1.1900 +/- 0.2044 | 1.1547 +/- 0.0962 |  |  |  | 0.2078 +/- 0.0051 |
| `mlp_h64_64` | 0.8203 +/- 0.1578 | 0.8948 +/- 0.0857 |  |  |  | 0.3410 +/- 0.0621 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75` | 0.4000 +/- 0.0791 | 0.5039 +/- 0.0753 |  |  | 128.0000 +/- 0.0000 | 1.4563 +/- 0.4128 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai8` | 0.5085 +/- 0.1110 | 0.7324 +/- 0.0351 |  |  | 128.0000 +/- 0.0000 | 0.9003 +/- 0.1497 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai16` | 0.6206 +/- 0.1190 | 0.8714 +/- 0.0684 |  |  | 128.0000 +/- 0.0000 | 1.2592 +/- 0.2324 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.3793 +/- 0.0771 | 0.4862 +/- 0.0728 |  |  | 128.0000 +/- 0.0000 | 1.1485 +/- 0.0921 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai8` | 0.5019 +/- 0.1094 | 0.7380 +/- 0.0352 |  |  | 128.0000 +/- 0.0000 | 1.1385 +/- 0.3496 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai16` | 0.6257 +/- 0.1192 | 0.8834 +/- 0.0685 |  |  | 128.0000 +/- 0.0000 | 1.1471 +/- 0.1059 |

`final_window_mse` best-kernel-vs-best-MLP diff: +0.4410 +/- 0.0810; wins/losses/ties 3/0/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 3}.

## Interpretation Bar

A positive result here is still a search result unless one fixed kernel configuration beats the best fair MLP across the broad suite. The `best_kernel_vs_best_mlp` rows are useful for detecting whether the mathematical mechanism has headroom; a universal learner claim requires promoting one canonical configuration and rerunning it without per-dataset selection.

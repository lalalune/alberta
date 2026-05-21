# D07 Budgeted Diffusion-KRLS Results

Protocol: 5 paired seeds, 2500 online steps, final window 500. Kernel=algebraic_green, updates=rls, budgets=[128], sigmas=[0.5], rhos=[1.0], novelty thresholds=[0.001].

This is a single-learner test. The kernel methods do not consume MLP predictions, routes, stacker weights, or offline labels. Positive kernel-vs-MLP paired differences favor the kernel method.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0914 +/- 0.0059 | 0.3427 +/- 0.0102 |  |  |  | 0.4403 +/- 0.1026 |
| `mlp_h128` | 0.0921 +/- 0.0102 | 0.3553 +/- 0.0151 |  |  |  | 0.4815 +/- 0.0986 |
| `mlp_h64_64` | 0.0804 +/- 0.0054 | 0.2514 +/- 0.0131 |  |  |  | 0.5397 +/- 0.1032 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.8294 +/- 0.0230 | 0.8730 +/- 0.0145 |  |  | 128.0000 +/- 0.0000 | 12.3253 +/- 1.9607 |

`final_window_mse` best-kernel-vs-best-MLP diff: -0.7570 +/- 0.0250; wins/losses/ties 0/5/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 5}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2656 +/- 0.0179 | 0.4978 +/- 0.0164 |  |  |  | 0.3182 +/- 0.0347 |
| `mlp_h128` | 0.4209 +/- 0.0158 | 0.5758 +/- 0.0148 |  |  |  | 0.3121 +/- 0.0356 |
| `mlp_h64_64` | 0.5773 +/- 0.0213 | 0.6757 +/- 0.0175 |  |  |  | 0.3748 +/- 0.0504 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.7667 +/- 0.0709 | 0.8175 +/- 0.0374 |  |  | 128.0000 +/- 0.0000 | 7.5201 +/- 1.7903 |

`final_window_mse` best-kernel-vs-best-MLP diff: -0.5011 +/- 0.0602; wins/losses/ties 0/5/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 5}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0415 +/- 0.0014 | 0.0814 +/- 0.0023 |  |  |  | 0.7884 +/- 0.4695 |
| `mlp_h128` | 0.0615 +/- 0.0029 | 0.0957 +/- 0.0027 |  |  |  | 0.6103 +/- 0.3366 |
| `mlp_h64_64` | 0.0700 +/- 0.0040 | 0.1069 +/- 0.0043 |  |  |  | 0.6494 +/- 0.2379 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.1547 +/- 0.0057 | 0.1606 +/- 0.0084 |  |  | 128.0000 +/- 0.0000 | 8.2536 +/- 0.5284 |

`final_window_mse` best-kernel-vs-best-MLP diff: -0.1132 +/- 0.0053; wins/losses/ties 0/5/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 5}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.7057 +/- 0.1117 | 0.9146 +/- 0.0547 |  |  |  | 0.1624 +/- 0.0142 |
| `mlp_h128` | 0.8194 +/- 0.1300 | 1.0028 +/- 0.0574 |  |  |  | 0.2080 +/- 0.0521 |
| `mlp_h64_64` | 1.0102 +/- 0.1588 | 1.0952 +/- 0.0686 |  |  |  | 0.3863 +/- 0.1285 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 1.7567 +/- 0.2382 | 1.8298 +/- 0.1725 |  |  | 128.0000 +/- 0.0000 | 4.6706 +/- 1.0064 |

`final_window_mse` best-kernel-vs-best-MLP diff: -1.0511 +/- 0.1307; wins/losses/ties 0/5/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 5}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.2174 +/- 0.1185 | 1.1402 +/- 0.0540 |  |  |  | 0.3393 +/- 0.0747 |
| `mlp_h128` | 1.2431 +/- 0.1287 | 1.1529 +/- 0.0556 |  |  |  | 0.3309 +/- 0.0856 |
| `mlp_h64_64` | 0.8693 +/- 0.1035 | 0.8904 +/- 0.0478 |  |  |  | 0.4959 +/- 0.1162 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 2.0638 +/- 0.3272 | 1.9013 +/- 0.2218 |  |  | 128.0000 +/- 0.0000 | 12.7973 +/- 3.1745 |

`final_window_mse` best-kernel-vs-best-MLP diff: -1.1945 +/- 0.2344; wins/losses/ties 0/5/0; best-kernel counts {'algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75': 5}.

## Interpretation Bar

A positive result here is still a search result unless one fixed kernel configuration beats the best fair MLP across the broad suite. The `best_kernel_vs_best_mlp` rows are useful for detecting whether the mathematical mechanism has headroom; a universal learner claim requires promoting one canonical configuration and rerunning it without per-dataset selection.

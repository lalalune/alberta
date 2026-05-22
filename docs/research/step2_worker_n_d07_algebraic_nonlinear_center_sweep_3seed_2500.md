# D07 Budgeted Diffusion-KRLS Results

Protocol: 3 paired seeds, 2500 online steps, final window 500. Kernel=algebraic_green, updates=rls, budgets=[128], sigmas=[0.25, 0.5], rhos=[0.99, 0.995, 1.0], novelty thresholds=[0.001].

This is a single-learner test. The kernel methods do not consume MLP predictions, routes, stacker weights, or offline labels. Positive kernel-vs-MLP paired differences favor the kernel method.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0418 +/- 0.0023 | 0.0791 +/- 0.0017 |  |  |  | 0.9958 +/- 0.6602 |
| `mlp_h128` | 0.0617 +/- 0.0038 | 0.0953 +/- 0.0021 |  |  |  | 0.5930 +/- 0.3678 |
| `mlp_h64_64` | 0.0697 +/- 0.0067 | 0.1039 +/- 0.0042 |  |  |  | 0.5628 +/- 0.2077 |
| `algebraic_green_rls_b128_s0p25_r0p99_n0p001_d3_aw0p75` | 0.1533 +/- 0.0062 | 0.1488 +/- 0.0098 |  |  | 128.0000 +/- 0.0000 | 9.4703 +/- 3.2552 |
| `algebraic_green_rls_b128_s0p25_r0p99_n0p001_d3_aw0p75_ai4` | 0.1556 +/- 0.0165 | 0.1502 +/- 0.0121 |  |  | 128.0000 +/- 0.0000 | 3.1378 +/- 1.0382 |
| `algebraic_green_rls_b128_s0p25_r0p99_n0p001_d3_aw0p75_ai8` | 0.1620 +/- 0.0133 | 0.1431 +/- 0.0141 |  |  | 128.0000 +/- 0.0000 | 1.8680 +/- 0.2460 |
| `algebraic_green_rls_b128_s0p25_r0p99_n0p001_d3_aw0p75_ai16` | 0.1566 +/- 0.0107 | 0.1215 +/- 0.0075 |  |  | 128.0000 +/- 0.0000 | 1.3613 +/- 0.3203 |
| `algebraic_green_rls_b128_s0p25_r0p995_n0p001_d3_aw0p75` | 0.1533 +/- 0.0062 | 0.1488 +/- 0.0098 |  |  | 128.0000 +/- 0.0000 | 7.0475 +/- 0.5803 |
| `algebraic_green_rls_b128_s0p25_r0p995_n0p001_d3_aw0p75_ai4` | 0.1648 +/- 0.0183 | 0.1603 +/- 0.0135 |  |  | 128.0000 +/- 0.0000 | 4.9553 +/- 2.3388 |
| `algebraic_green_rls_b128_s0p25_r0p995_n0p001_d3_aw0p75_ai8` | 0.1899 +/- 0.0208 | 0.1609 +/- 0.0163 |  |  | 128.0000 +/- 0.0000 | 2.4089 +/- 0.6780 |
| `algebraic_green_rls_b128_s0p25_r0p995_n0p001_d3_aw0p75_ai16` | 0.1507 +/- 0.0153 | 0.1237 +/- 0.0087 |  |  | 128.0000 +/- 0.0000 | 2.2530 +/- 0.6284 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75` | 0.1533 +/- 0.0062 | 0.1488 +/- 0.0098 |  |  | 128.0000 +/- 0.0000 | 9.1584 +/- 1.7544 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai4` | 0.1729 +/- 0.0187 | 0.1688 +/- 0.0136 |  |  | 128.0000 +/- 0.0000 | 3.9464 +/- 1.0623 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai8` | 0.2038 +/- 0.0206 | 0.1767 +/- 0.0151 |  |  | 128.0000 +/- 0.0000 | 2.8866 +/- 1.5317 |
| `algebraic_green_rls_b128_s0p25_r1_n0p001_d3_aw0p75_ai16` | 0.2079 +/- 0.0277 | 0.1476 +/- 0.0125 |  |  | 128.0000 +/- 0.0000 | 2.3346 +/- 1.3413 |
| `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75` | 0.1555 +/- 0.0056 | 0.1518 +/- 0.0111 |  |  | 128.0000 +/- 0.0000 | 10.8259 +/- 3.5039 |
| `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai4` | 0.1567 +/- 0.0169 | 0.1536 +/- 0.0131 |  |  | 128.0000 +/- 0.0000 | 3.4624 +/- 0.4898 |
| `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai8` | 0.1580 +/- 0.0132 | 0.1417 +/- 0.0148 |  |  | 128.0000 +/- 0.0000 | 1.9706 +/- 0.7888 |
| `algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai16` | 0.1391 +/- 0.0082 | 0.1162 +/- 0.0069 |  |  | 128.0000 +/- 0.0000 | 2.5305 +/- 0.4157 |
| `algebraic_green_rls_b128_s0p5_r0p995_n0p001_d3_aw0p75` | 0.1555 +/- 0.0056 | 0.1518 +/- 0.0111 |  |  | 128.0000 +/- 0.0000 | 8.9628 +/- 0.9972 |
| `algebraic_green_rls_b128_s0p5_r0p995_n0p001_d3_aw0p75_ai4` | 0.1681 +/- 0.0194 | 0.1654 +/- 0.0150 |  |  | 128.0000 +/- 0.0000 | 2.9952 +/- 0.4199 |
| `algebraic_green_rls_b128_s0p5_r0p995_n0p001_d3_aw0p75_ai8` | 0.1842 +/- 0.0181 | 0.1611 +/- 0.0150 |  |  | 128.0000 +/- 0.0000 | 2.5137 +/- 0.2401 |
| `algebraic_green_rls_b128_s0p5_r0p995_n0p001_d3_aw0p75_ai16` | 0.1471 +/- 0.0145 | 0.1226 +/- 0.0085 |  |  | 128.0000 +/- 0.0000 | 1.8604 +/- 0.1702 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75` | 0.1555 +/- 0.0056 | 0.1518 +/- 0.0111 |  |  | 128.0000 +/- 0.0000 | 10.7506 +/- 0.3115 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai4` | 0.1745 +/- 0.0200 | 0.1723 +/- 0.0144 |  |  | 128.0000 +/- 0.0000 | 3.5293 +/- 0.6159 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai8` | 0.2032 +/- 0.0205 | 0.1791 +/- 0.0145 |  |  | 128.0000 +/- 0.0000 | 2.1689 +/- 0.0479 |
| `algebraic_green_rls_b128_s0p5_r1_n0p001_d3_aw0p75_ai16` | 0.2129 +/- 0.0270 | 0.1488 +/- 0.0124 |  |  | 128.0000 +/- 0.0000 | 1.0073 +/- 0.1347 |

`final_window_mse` best-kernel-vs-best-MLP diff: -0.0960 +/- 0.0085; wins/losses/ties 0/3/0; best-kernel counts {'algebraic_green_rls_b128_s0p25_r0p995_n0p001_d3_aw0p75_ai16': 1, 'algebraic_green_rls_b128_s0p25_r0p99_n0p001_d3_aw0p75_ai4': 1, 'algebraic_green_rls_b128_s0p5_r0p99_n0p001_d3_aw0p75_ai16': 1}.

## Interpretation Bar

A positive result here is still a search result unless one fixed kernel configuration beats the best fair MLP across the broad suite. The `best_kernel_vs_best_mlp` rows are useful for detecting whether the mathematical mechanism has headroom; a universal learner claim requires promoting one canonical configuration and rerunning it without per-dataset selection.

# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p5_basis_0p4.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier basis. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 0.4347 +/- 0.1127 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 0.4653 +/- 0.0766 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 0.5460 +/- 0.0137 |
| `d18_core_0p5_basis_0p4` | 0.0329 +/- 0.0008 | 0.1147 +/- 0.0010 |  |  | 320.0000 +/- 0.0000 | 6.7720 +/- 1.0350 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1240 +/- 0.0265; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.4478 +/- 0.0517 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.4590 +/- 0.1541 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 0.5047 +/- 0.1226 |
| `d18_core_0p5_basis_0p4` | 0.0203 +/- 0.0018 | 0.1266 +/- 0.0144 |  |  | 320.0000 +/- 0.0000 | 4.2424 +/- 0.4574 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4079 +/- 0.0225; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.3996 +/- 0.0799 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 1.3646 +/- 0.5853 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 1.1054 +/- 0.2139 |
| `d18_core_0p5_basis_0p4` | 0.0320 +/- 0.0006 | 0.0457 +/- 0.0006 | 0.8978 +/- 0.0022 | 0.9592 +/- 0.0039 | 320.0000 +/- 0.0000 | 5.4885 +/- 0.8921 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0061 +/- 0.0011; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0519 +/- 0.0054; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.8629 +/- 0.0705 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 0.9919 +/- 0.2220 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.9226 +/- 0.0943 |
| `d18_core_0p5_basis_0p4` | 0.0451 +/- 0.0013 | 0.0531 +/- 0.0013 | 0.8078 +/- 0.0178 | 0.8466 +/- 0.0163 | 320.0000 +/- 0.0000 | 5.4730 +/- 0.6025 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0028 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0266 +/- 0.0101; wins/losses/ties 3/0/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 1.3524 +/- 0.2030 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 1.3387 +/- 0.2124 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 1.4416 +/- 0.0921 |
| `d18_core_0p5_basis_0p4` | 0.0381 +/- 0.0014 | 0.0458 +/- 0.0003 | 0.8844 +/- 0.0097 | 0.9264 +/- 0.0087 | 320.0000 +/- 0.0000 | 7.7221 +/- 0.6389 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0110 +/- 0.0011; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0451 +/- 0.0051; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.5231 +/- 0.6267 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.1944 +/- 0.3528 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.8750 +/- 0.2406 |
| `d18_core_0p5_basis_0p4` | 0.2411 +/- 0.0642 | 0.1554 +/- 0.0233 |  |  | 320.0000 +/- 0.0000 | 8.2664 +/- 1.2021 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0306 +/- 0.0314; wins/losses/ties 2/1/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.8547 +/- 0.1888 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 1.0519 +/- 0.2198 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.9804 +/- 0.4438 |
| `d18_core_0p5_basis_0p4` | 0.9093 +/- 0.1731 | 1.0018 +/- 0.2156 |  |  | 320.0000 +/- 0.0000 | 6.3839 +/- 2.4190 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2392 +/- 0.0958; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.3842 +/- 0.9183 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.9908 +/- 0.5790 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.8510 +/- 0.1917 |
| `d18_core_0p5_basis_0p4` | 1.0385 +/- 0.5953 | 1.1072 +/- 0.2793 |  |  | 320.0000 +/- 0.0000 | 4.0819 +/- 1.0213 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0910 +/- 0.0492; wins/losses/ties 1/2/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.

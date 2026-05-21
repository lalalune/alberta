# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_lowcore_poly_unified.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 2.1283 +/- 1.2934 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 2.2454 +/- 1.2908 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 1.5217 +/- 0.3448 |
| `d18_gain_lowcore_poly_unified` | 0.0422 +/- 0.0014 | 0.0848 +/- 0.0017 |  |  | 320.0000 +/- 0.0000 | 17.8982 +/- 3.3538 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1146 +/- 0.0274; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 1.5196 +/- 0.5167 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 1.6996 +/- 0.1663 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 2.1568 +/- 0.3880 |
| `d18_gain_lowcore_poly_unified` | 0.0262 +/- 0.0027 | 0.1413 +/- 0.0167 |  |  | 320.0000 +/- 0.0000 | 31.0130 +/- 4.3365 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4021 +/- 0.0221; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 10.2410 +/- 0.0362 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 10.1328 +/- 2.7082 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 9.8916 +/- 0.9105 |
| `d18_gain_lowcore_poly_unified` | 0.0445 +/- 0.0001 | 0.0524 +/- 0.0007 | 0.8178 +/- 0.0106 | 0.9208 +/- 0.0031 | 320.0000 +/- 0.0000 | 46.2563 +/- 6.7181 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0064 +/- 0.0010; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0136 +/- 0.0027; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 4.5377 +/- 1.5568 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 3.1805 +/- 0.6589 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 6.1365 +/- 2.5919 |
| `d18_gain_lowcore_poly_unified` | 0.0479 +/- 0.0015 | 0.0565 +/- 0.0019 | 0.7689 +/- 0.0318 | 0.8083 +/- 0.0150 | 320.0000 +/- 0.0000 | 25.5100 +/- 4.8463 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0001 +/- 0.0009; wins/losses/ties 1/2/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0118 +/- 0.0182; wins/losses/ties 1/2/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 3.4964 +/- 0.8817 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 2.4116 +/- 0.4825 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 2.6400 +/- 0.2006 |
| `d18_gain_lowcore_poly_unified` | 0.0510 +/- 0.0043 | 0.0528 +/- 0.0013 | 0.8489 +/- 0.0198 | 0.8881 +/- 0.0193 | 320.0000 +/- 0.0000 | 27.2888 +/- 3.8158 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0019 +/- 0.0020; wins/losses/ties 1/2/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0068 +/- 0.0086; wins/losses/ties 2/1/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.7019 +/- 1.1173 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.1211 +/- 0.4625 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.6456 +/- 0.0760 |
| `d18_gain_lowcore_poly_unified` | 0.2367 +/- 0.0792 | 0.1478 +/- 0.0260 |  |  | 320.0000 +/- 0.0000 | 8.4675 +/- 1.4089 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0350 +/- 0.0167; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.7454 +/- 0.1149 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 1.2061 +/- 0.3109 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 2.4625 +/- 0.8018 |
| `d18_gain_lowcore_poly_unified` | 0.9268 +/- 0.1816 | 0.9217 +/- 0.1932 |  |  | 320.0000 +/- 0.0000 | 12.1692 +/- 2.2583 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2217 +/- 0.1337; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 2.9819 +/- 2.2842 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 2.1725 +/- 1.2902 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 1.4575 +/- 0.2702 |
| `d18_gain_lowcore_poly_unified` | 0.6776 +/- 0.3906 | 0.6733 +/- 0.1502 |  |  | 320.0000 +/- 0.0000 | 19.0389 +/- 2.2544 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2699 +/- 0.2538; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

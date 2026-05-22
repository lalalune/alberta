# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 2.7355 +/- 1.9184 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 3.1972 +/- 2.4631 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 3.0094 +/- 1.7498 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0388 +/- 0.0006 | 0.0858 +/- 0.0011 |  |  | 320.0000 +/- 0.0000 | 20.0334 +/- 3.1201 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1181 +/- 0.0268; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.6717 +/- 0.1512 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.9319 +/- 0.1845 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 1.8613 +/- 0.5568 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0303 +/- 0.0041 | 0.0998 +/- 0.0107 |  |  | 320.0000 +/- 0.0000 | 24.9579 +/- 5.4080 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3979 +/- 0.0211; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 4.1725 +/- 0.5868 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 5.4553 +/- 2.2139 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 5.0344 +/- 0.6906 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0348 +/- 0.0004 | 0.0464 +/- 0.0006 | 0.8744 +/- 0.0089 | 0.9536 +/- 0.0095 | 320.0000 +/- 0.0000 | 31.1535 +/- 1.2592 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0033 +/- 0.0010; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0464 +/- 0.0102; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 2.8527 +/- 0.6583 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 2.6966 +/- 0.5224 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 3.8260 +/- 1.5294 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0448 +/- 0.0016 | 0.0528 +/- 0.0013 | 0.7800 +/- 0.0231 | 0.8275 +/- 0.0193 | 320.0000 +/- 0.0000 | 19.9906 +/- 2.2361 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0031 +/- 0.0009; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0074 +/- 0.0092; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 1.3954 +/- 0.0423 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 1.4139 +/- 0.1181 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 1.6992 +/- 0.3555 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0379 +/- 0.0013 | 0.0458 +/- 0.0002 | 0.8744 +/- 0.0122 | 0.9252 +/- 0.0094 | 320.0000 +/- 0.0000 | 15.3489 +/- 1.4341 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0113 +/- 0.0013; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0439 +/- 0.0062; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 3.4454 +/- 1.4142 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 3.7909 +/- 1.3894 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 2.6079 +/- 0.2022 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2401 +/- 0.0772 | 0.1518 +/- 0.0261 |  |  | 320.0000 +/- 0.0000 | 27.0369 +/- 3.8276 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0316 +/- 0.0184; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.7407 +/- 0.1639 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.7133 +/- 0.1928 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 1.0829 +/- 0.3582 |
| `d18_gain_safecore_poly_unified_0p01` | 0.8825 +/- 0.1989 | 0.8828 +/- 0.1736 |  |  | 320.0000 +/- 0.0000 | 14.1624 +/- 1.8484 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2659 +/- 0.0783; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 3.5715 +/- 1.1072 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 3.5803 +/- 1.2116 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 2.2893 +/- 0.7323 |
| `d18_gain_safecore_poly_unified_0p01` | 0.7981 +/- 0.5213 | 0.8066 +/- 0.2051 |  |  | 320.0000 +/- 0.0000 | 28.9554 +/- 3.4590 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1494 +/- 0.1232; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

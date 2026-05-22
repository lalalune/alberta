# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.6760 +/- 0.3127 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.5843 +/- 0.1772 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.5468 +/- 0.0583 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0357 +/- 0.0114 | 0.0654 +/- 0.0170 |  |  | 320.0000 +/- 0.0000 | 4.2504 +/- 1.0344 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0374 +/- 0.0022; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 0.7889 +/- 0.0628 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 0.9333 +/- 0.1355 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 1.2236 +/- 0.3241 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0055 +/- 0.0002 | 0.0047 +/- 0.0001 | 0.9844 +/- 0.0011 | 0.2492 +/- 0.0181 | 320.0000 +/- 0.0000 | 6.4962 +/- 0.7310 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0026 +/- 0.0002; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0835 +/- 0.0130; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 1.8719 +/- 0.8078 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 2.0372 +/- 1.0244 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 1.2869 +/- 0.3269 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0451 +/- 0.0012 | 0.0532 +/- 0.0013 | 0.8078 +/- 0.0179 | 0.8479 +/- 0.0162 | 320.0000 +/- 0.0000 | 6.2205 +/- 0.2798 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0027 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0278 +/- 0.0095; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.2691 +/- 0.8377 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.1336 +/- 0.4993 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.7350 +/- 0.1605 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2398 +/- 0.0638 | 0.1544 +/- 0.0232 |  |  | 320.0000 +/- 0.0000 | 6.1009 +/- 1.5322 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0319 +/- 0.0317; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.3883 +/- 0.0545 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.5392 +/- 0.0511 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.5311 +/- 0.0610 |
| `d18_gain_safecore_poly_unified_0p01` | 0.9103 +/- 0.1736 | 1.0032 +/- 0.2159 |  |  | 320.0000 +/- 0.0000 | 3.5382 +/- 0.3082 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2381 +/- 0.0952; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.8393 +/- 0.4127 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.9579 +/- 0.4828 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.5361 +/- 0.0886 |
| `d18_gain_safecore_poly_unified_0p01` | 1.0348 +/- 0.5931 | 1.1024 +/- 0.2787 |  |  | 320.0000 +/- 0.0000 | 4.2198 +/- 0.5478 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0873 +/- 0.0514; wins/losses/ties 1/2/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 6.9898 +/- 3.5179 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 5.6621 +/- 3.0405 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 4.8081 +/- 1.2458 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0348 +/- 0.0004 | 0.0464 +/- 0.0006 | 0.8744 +/- 0.0089 | 0.9536 +/- 0.0095 | 320.0000 +/- 0.0000 | 24.0165 +/- 2.4267 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0033 +/- 0.0010; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0464 +/- 0.0102; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 6.6843 +/- 0.5201 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 4.4310 +/- 0.8205 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 4.5042 +/- 1.4355 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0448 +/- 0.0016 | 0.0528 +/- 0.0013 | 0.7800 +/- 0.0231 | 0.8275 +/- 0.0193 | 320.0000 +/- 0.0000 | 34.7382 +/- 6.0210 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0031 +/- 0.0009; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0074 +/- 0.0092; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 5.6248 +/- 2.9984 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 4.2458 +/- 2.0082 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 4.1079 +/- 1.7307 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0379 +/- 0.0013 | 0.0458 +/- 0.0002 | 0.8744 +/- 0.0122 | 0.9252 +/- 0.0094 | 320.0000 +/- 0.0000 | 15.9619 +/- 1.5490 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0113 +/- 0.0013; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0439 +/- 0.0062; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 2.1015 +/- 0.8205 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 3.0838 +/- 1.5044 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.7830 +/- 0.2916 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2401 +/- 0.0772 | 0.1518 +/- 0.0261 |  |  | 320.0000 +/- 0.0000 | 21.5186 +/- 5.2478 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0316 +/- 0.0184; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 3.1527 +/- 1.6646 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 3.3854 +/- 1.6273 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 1.7744 +/- 0.1726 |
| `d18_gain_safecore_poly_unified_0p01` | 0.7981 +/- 0.5213 | 0.8066 +/- 0.2051 |  |  | 320.0000 +/- 0.0000 | 29.1476 +/- 3.8872 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1494 +/- 0.1232; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

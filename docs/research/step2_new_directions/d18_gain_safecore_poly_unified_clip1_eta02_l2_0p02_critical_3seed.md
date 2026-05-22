# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p005, d18_gain_safecore_poly_unified_0p01, d18_gain_safecore_poly_unified_0p02.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.3678 +/- 0.5592 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 1.4935 +/- 0.5999 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 1.4372 +/- 0.4241 |
| `d18_gain_safecore_poly_unified_0p005` | 0.0417 +/- 0.0004 | 0.0494 +/- 0.0007 | 0.8056 +/- 0.0062 | 0.9295 +/- 0.0152 | 320.0000 +/- 0.0000 | 7.3903 +/- 1.2789 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0416 +/- 0.0004 | 0.0494 +/- 0.0007 | 0.8089 +/- 0.0022 | 0.9314 +/- 0.0150 | 320.0000 +/- 0.0000 | 7.8934 +/- 0.9860 |
| `d18_gain_safecore_poly_unified_0p02` | 0.0417 +/- 0.0003 | 0.0494 +/- 0.0007 | 0.8133 +/- 0.0033 | 0.9314 +/- 0.0142 | 320.0000 +/- 0.0000 | 7.9216 +/- 2.5104 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0035 +/- 0.0007; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 2, 'd18_gain_safecore_poly_unified_0p02': 1}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0247 +/- 0.0138; wins/losses/ties 2/1/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 1.3957 +/- 0.5895 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 0.9563 +/- 0.3373 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 1.0557 +/- 0.4268 |
| `d18_gain_safecore_poly_unified_0p005` | 0.0466 +/- 0.0021 | 0.0536 +/- 0.0012 | 0.7578 +/- 0.0301 | 0.7885 +/- 0.0132 | 320.0000 +/- 0.0000 | 4.0620 +/- 0.1389 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0466 +/- 0.0020 | 0.0536 +/- 0.0012 | 0.7633 +/- 0.0265 | 0.7879 +/- 0.0192 | 320.0000 +/- 0.0000 | 4.9266 +/- 1.1592 |
| `d18_gain_safecore_poly_unified_0p02` | 0.0467 +/- 0.0020 | 0.0536 +/- 0.0011 | 0.7611 +/- 0.0294 | 0.7972 +/- 0.0194 | 320.0000 +/- 0.0000 | 3.7168 +/- 0.2990 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0013 +/- 0.0013; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p005': 1, 'd18_gain_safecore_poly_unified_0p01': 2}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0216 +/- 0.0140; wins/losses/ties 1/2/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 0.5808 +/- 0.0274 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 0.9239 +/- 0.4311 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 0.6167 +/- 0.0110 |
| `d18_gain_safecore_poly_unified_0p005` | 0.0420 +/- 0.0015 | 0.0475 +/- 0.0003 | 0.8378 +/- 0.0174 | 0.8893 +/- 0.0152 | 320.0000 +/- 0.0000 | 4.9010 +/- 0.9510 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0420 +/- 0.0015 | 0.0475 +/- 0.0003 | 0.8378 +/- 0.0174 | 0.8862 +/- 0.0150 | 320.0000 +/- 0.0000 | 3.5256 +/- 0.6690 |
| `d18_gain_safecore_poly_unified_0p02` | 0.0419 +/- 0.0015 | 0.0475 +/- 0.0002 | 0.8400 +/- 0.0145 | 0.8881 +/- 0.0132 | 320.0000 +/- 0.0000 | 5.5219 +/- 1.1547 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0073 +/- 0.0010; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p005': 1, 'd18_gain_safecore_poly_unified_0p02': 2}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0087 +/- 0.0082; wins/losses/ties 2/1/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 2.1101 +/- 1.0081 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.8839 +/- 0.9449 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.1302 +/- 0.1543 |
| `d18_gain_safecore_poly_unified_0p005` | 0.2442 +/- 0.0824 | 0.1521 +/- 0.0268 |  |  | 320.0000 +/- 0.0000 | 14.4437 +/- 2.4819 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2433 +/- 0.0784 | 0.1522 +/- 0.0257 |  |  | 320.0000 +/- 0.0000 | 13.2110 +/- 3.8207 |
| `d18_gain_safecore_poly_unified_0p02` | 0.2424 +/- 0.0808 | 0.1514 +/- 0.0261 |  |  | 320.0000 +/- 0.0000 | 14.5821 +/- 3.5628 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0314 +/- 0.0158; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p005': 1, 'd18_gain_safecore_poly_unified_0p01': 1, 'd18_gain_safecore_poly_unified_0p02': 1}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 3.7342 +/- 0.7141 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 3.8362 +/- 0.9938 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 3.0445 +/- 1.0074 |
| `d18_gain_safecore_poly_unified_0p005` | 0.7624 +/- 0.5057 | 0.7668 +/- 0.1961 |  |  | 320.0000 +/- 0.0000 | 37.2251 +/- 10.2806 |
| `d18_gain_safecore_poly_unified_0p01` | 0.7599 +/- 0.5045 | 0.7663 +/- 0.1968 |  |  | 320.0000 +/- 0.0000 | 31.9952 +/- 6.0631 |
| `d18_gain_safecore_poly_unified_0p02` | 0.7614 +/- 0.5054 | 0.7644 +/- 0.1951 |  |  | 320.0000 +/- 0.0000 | 28.8409 +/- 4.2147 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1876 +/- 0.1401; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

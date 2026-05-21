# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_fixedcore_poly_unified_0p005, d18_gain_fixedcore_poly_unified_0p01, d18_gain_fixedcore_poly_unified_0p02.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.7972 +/- 0.7499 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 2.0602 +/- 0.8165 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 1.1448 +/- 0.2893 |
| `d18_gain_fixedcore_poly_unified_0p005` | 0.0365 +/- 0.0005 | 0.0527 +/- 0.0021 | 0.8878 +/- 0.0080 | 0.9419 +/- 0.0118 | 320.0000 +/- 0.0000 | 6.4036 +/- 0.4072 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0366 +/- 0.0006 | 0.0528 +/- 0.0022 | 0.8889 +/- 0.0106 | 0.9400 +/- 0.0125 | 320.0000 +/- 0.0000 | 12.2264 +/- 2.2301 |
| `d18_gain_fixedcore_poly_unified_0p02` | 0.0365 +/- 0.0005 | 0.0528 +/- 0.0022 | 0.8878 +/- 0.0080 | 0.9412 +/- 0.0118 | 320.0000 +/- 0.0000 | 7.3202 +/- 1.3750 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0017 +/- 0.0006; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p005': 1, 'd18_gain_fixedcore_poly_unified_0p01': 2}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0346 +/- 0.0118; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 2.5942 +/- 0.7975 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 1.9696 +/- 0.6633 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 2.1755 +/- 0.5994 |
| `d18_gain_fixedcore_poly_unified_0p005` | 0.0537 +/- 0.0034 | 0.0658 +/- 0.0065 | 0.7800 +/- 0.0033 | 0.8262 +/- 0.0077 | 320.0000 +/- 0.0000 | 14.5223 +/- 4.0130 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0539 +/- 0.0036 | 0.0663 +/- 0.0069 | 0.7700 +/- 0.0033 | 0.8207 +/- 0.0116 | 320.0000 +/- 0.0000 | 16.4741 +/- 3.7193 |
| `d18_gain_fixedcore_poly_unified_0p02` | 0.0543 +/- 0.0033 | 0.0663 +/- 0.0071 | 0.7800 +/- 0.0033 | 0.8207 +/- 0.0087 | 320.0000 +/- 0.0000 | 21.8233 +/- 2.7892 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0059 +/- 0.0042; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p005': 2, 'd18_gain_fixedcore_poly_unified_0p01': 1}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0062 +/- 0.0139; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 2.2676 +/- 0.7149 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 1.7363 +/- 0.2367 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 1.7971 +/- 0.2411 |
| `d18_gain_fixedcore_poly_unified_0p005` | 0.1162 +/- 0.0318 | 0.0743 +/- 0.0098 | 0.8222 +/- 0.0337 | 0.8627 +/- 0.0407 | 320.0000 +/- 0.0000 | 15.1777 +/- 3.4348 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.1294 +/- 0.0455 | 0.0781 +/- 0.0132 | 0.8211 +/- 0.0327 | 0.8553 +/- 0.0435 | 320.0000 +/- 0.0000 | 14.7657 +/- 2.0262 |
| `d18_gain_fixedcore_poly_unified_0p02` | 0.1292 +/- 0.0464 | 0.0776 +/- 0.0133 | 0.8111 +/- 0.0290 | 0.8621 +/- 0.0439 | 320.0000 +/- 0.0000 | 13.5580 +/- 2.2717 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0637 +/- 0.0319; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p005': 1, 'd18_gain_fixedcore_poly_unified_0p01': 1, 'd18_gain_fixedcore_poly_unified_0p02': 1}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0167 +/- 0.0340; wins/losses/ties 1/2/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 4.5945 +/- 2.3719 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 3.3485 +/- 1.1151 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.6920 +/- 0.1731 |
| `d18_gain_fixedcore_poly_unified_0p005` | 0.2581 +/- 0.0867 | 0.1602 +/- 0.0280 |  |  | 320.0000 +/- 0.0000 | 18.0477 +/- 0.7138 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.2563 +/- 0.0848 | 0.1601 +/- 0.0271 |  |  | 320.0000 +/- 0.0000 | 19.6751 +/- 1.1763 |
| `d18_gain_fixedcore_poly_unified_0p02` | 0.2570 +/- 0.0868 | 0.1597 +/- 0.0275 |  |  | 320.0000 +/- 0.0000 | 23.1928 +/- 1.9677 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0167 +/- 0.0110; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p005': 1, 'd18_gain_fixedcore_poly_unified_0p01': 1, 'd18_gain_fixedcore_poly_unified_0p02': 1}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 4.5455 +/- 3.0402 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 4.8552 +/- 3.0440 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 2.3436 +/- 0.3580 |
| `d18_gain_fixedcore_poly_unified_0p005` | 0.8840 +/- 0.4332 | 0.8056 +/- 0.1887 |  |  | 320.0000 +/- 0.0000 | 29.0633 +/- 3.5903 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.8815 +/- 0.4314 | 0.8008 +/- 0.1870 |  |  | 320.0000 +/- 0.0000 | 25.9784 +/- 1.8903 |
| `d18_gain_fixedcore_poly_unified_0p02` | 0.8814 +/- 0.4311 | 0.8017 +/- 0.1880 |  |  | 320.0000 +/- 0.0000 | 26.8774 +/- 3.2017 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0662 +/- 0.2150; wins/losses/ties 1/2/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 2, 'd18_gain_fixedcore_poly_unified_0p02': 1}.

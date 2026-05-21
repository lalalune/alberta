# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_fixedcore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 10.7267 +/- 4.8559 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 4.8552 +/- 1.6356 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 3.4286 +/- 0.6052 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0330 +/- 0.0005 | 0.0469 +/- 0.0006 | 0.9033 +/- 0.0051 | 0.9536 +/- 0.0065 | 320.0000 +/- 0.0000 | 34.7422 +/- 7.3700 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0051 +/- 0.0011; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0464 +/- 0.0077; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 5.0151 +/- 0.9945 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 6.1584 +/- 1.2704 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 7.3071 +/- 1.9348 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0463 +/- 0.0012 | 0.0544 +/- 0.0014 | 0.7944 +/- 0.0197 | 0.8429 +/- 0.0139 | 320.0000 +/- 0.0000 | 38.1550 +/- 4.6769 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0015 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0229 +/- 0.0080; wins/losses/ties 3/0/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 5.0722 +/- 0.9203 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 3.7292 +/- 0.7444 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 4.5604 +/- 1.0950 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0398 +/- 0.0015 | 0.0475 +/- 0.0004 | 0.8844 +/- 0.0109 | 0.9270 +/- 0.0091 | 320.0000 +/- 0.0000 | 31.7304 +/- 3.8554 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0093 +/- 0.0009; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0458 +/- 0.0053; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 2.4199 +/- 1.0371 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 2.8363 +/- 1.9985 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 2.6731 +/- 1.1614 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.2345 +/- 0.0684 | 0.1511 +/- 0.0245 |  |  | 320.0000 +/- 0.0000 | 20.6325 +/- 3.9119 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0372 +/- 0.0270; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 3.6610 +/- 1.3719 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 3.0620 +/- 0.9951 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 2.1071 +/- 0.5227 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.8956 +/- 0.4937 | 0.8664 +/- 0.2084 |  |  | 320.0000 +/- 0.0000 | 20.8933 +/- 1.8799 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0519 +/- 0.1508; wins/losses/ties 1/2/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.

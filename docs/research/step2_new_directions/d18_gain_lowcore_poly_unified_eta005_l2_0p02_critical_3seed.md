# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_lowcore_poly_unified.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 5.1184 +/- 1.9045 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 3.8031 +/- 0.6349 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 5.5958 +/- 1.2003 |
| `d18_gain_lowcore_poly_unified` | 0.0415 +/- 0.0001 | 0.0517 +/- 0.0007 | 0.8400 +/- 0.0038 | 0.9283 +/- 0.0059 | 320.0000 +/- 0.0000 | 26.9770 +/- 4.5888 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0033 +/- 0.0010; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0210 +/- 0.0054; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 2.4470 +/- 0.4152 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 2.5866 +/- 0.7129 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 3.2184 +/- 0.6247 |
| `d18_gain_lowcore_poly_unified` | 0.0466 +/- 0.0015 | 0.0565 +/- 0.0021 | 0.7778 +/- 0.0270 | 0.8219 +/- 0.0116 | 320.0000 +/- 0.0000 | 24.5712 +/- 4.7722 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0012 +/- 0.0009; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0019 +/- 0.0168; wins/losses/ties 1/2/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 3.6695 +/- 0.2537 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 2.5888 +/- 0.3921 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 3.0422 +/- 0.7722 |
| `d18_gain_lowcore_poly_unified` | 0.0546 +/- 0.0055 | 0.0540 +/- 0.0016 | 0.8500 +/- 0.0173 | 0.8942 +/- 0.0225 | 320.0000 +/- 0.0000 | 22.9280 +/- 1.7945 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0055 +/- 0.0033; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0130 +/- 0.0118; wins/losses/ties 2/1/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.2679 +/- 0.5663 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.4652 +/- 0.7671 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.9173 +/- 0.2698 |
| `d18_gain_lowcore_poly_unified` | 0.2347 +/- 0.0780 | 0.1471 +/- 0.0258 |  |  | 320.0000 +/- 0.0000 | 9.2334 +/- 1.6681 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0371 +/- 0.0177; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.9983 +/- 0.4656 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.0974 +/- 0.4431 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.8819 +/- 0.1175 |
| `d18_gain_lowcore_poly_unified` | 0.6900 +/- 0.4008 | 0.6835 +/- 0.1513 |  |  | 320.0000 +/- 0.0000 | 9.3677 +/- 0.3758 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2575 +/- 0.2436; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

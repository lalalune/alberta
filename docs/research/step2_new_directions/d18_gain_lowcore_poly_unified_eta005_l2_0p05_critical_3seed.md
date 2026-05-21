# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_lowcore_poly_unified.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 5.1754 +/- 1.1271 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 6.2218 +/- 0.7157 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 3.7293 +/- 0.8756 |
| `d18_gain_lowcore_poly_unified` | 0.0400 +/- 0.0000 | 0.0516 +/- 0.0008 | 0.8600 +/- 0.0069 | 0.9382 +/- 0.0113 | 320.0000 +/- 0.0000 | 27.6164 +/- 4.5363 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0019 +/- 0.0010; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0309 +/- 0.0114; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 2.7381 +/- 0.4667 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 2.9655 +/- 0.4624 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 3.5431 +/- 0.7735 |
| `d18_gain_lowcore_poly_unified` | 0.0471 +/- 0.0012 | 0.0574 +/- 0.0024 | 0.7856 +/- 0.0157 | 0.8256 +/- 0.0102 | 320.0000 +/- 0.0000 | 21.8796 +/- 3.3917 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0007 +/- 0.0008; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0056 +/- 0.0130; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 3.5321 +/- 0.3840 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 2.6822 +/- 0.3801 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 3.1848 +/- 0.6325 |
| `d18_gain_lowcore_poly_unified` | 0.0603 +/- 0.0061 | 0.0559 +/- 0.0017 | 0.8467 +/- 0.0176 | 0.8868 +/- 0.0235 | 320.0000 +/- 0.0000 | 21.5486 +/- 2.3557 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0111 +/- 0.0040; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0056 +/- 0.0144; wins/losses/ties 1/2/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.2130 +/- 0.4960 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.1899 +/- 0.5424 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.9501 +/- 0.1011 |
| `d18_gain_lowcore_poly_unified` | 0.2324 +/- 0.0765 | 0.1464 +/- 0.0255 |  |  | 320.0000 +/- 0.0000 | 9.8681 +/- 1.5681 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0393 +/- 0.0191; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.0962 +/- 0.6653 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.0192 +/- 0.5202 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.5917 +/- 0.0929 |
| `d18_gain_lowcore_poly_unified` | 0.7209 +/- 0.4196 | 0.7065 +/- 0.1549 |  |  | 320.0000 +/- 0.0000 | 9.8217 +/- 0.6002 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2266 +/- 0.2249; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

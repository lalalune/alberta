# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_fixedcore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 6.4926 +/- 1.5099 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 6.5928 +/- 0.7773 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 7.8328 +/- 2.1150 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0333 +/- 0.0006 | 0.0474 +/- 0.0006 | 0.9044 +/- 0.0068 | 0.9518 +/- 0.0065 | 320.0000 +/- 0.0000 | 40.8248 +/- 1.1903 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0048 +/- 0.0011; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0445 +/- 0.0077; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 5.8359 +/- 1.7745 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 5.7621 +/- 0.9005 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 4.1336 +/- 0.2962 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0471 +/- 0.0014 | 0.0554 +/- 0.0014 | 0.7867 +/- 0.0164 | 0.8355 +/- 0.0215 | 320.0000 +/- 0.0000 | 20.9699 +/- 4.3846 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0007 +/- 0.0007; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0155 +/- 0.0097; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 2.5956 +/- 0.4379 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 2.4125 +/- 0.6197 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 2.5214 +/- 0.6108 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0401 +/- 0.0016 | 0.0478 +/- 0.0005 | 0.8833 +/- 0.0126 | 0.9233 +/- 0.0104 | 320.0000 +/- 0.0000 | 15.9921 +/- 2.1268 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0091 +/- 0.0008; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0421 +/- 0.0063; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.7394 +/- 1.2608 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.9668 +/- 0.5051 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.6227 +/- 0.1216 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.2399 +/- 0.0763 | 0.1524 +/- 0.0258 |  |  | 320.0000 +/- 0.0000 | 8.1047 +/- 1.2949 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0318 +/- 0.0192; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 2.3771 +/- 1.8815 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.0977 +/- 0.7027 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.6929 +/- 0.1721 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.9252 +/- 0.5187 | 0.9110 +/- 0.2328 |  |  | 320.0000 +/- 0.0000 | 9.6179 +/- 0.4814 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0223 +/- 0.1258; wins/losses/ties 1/2/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.

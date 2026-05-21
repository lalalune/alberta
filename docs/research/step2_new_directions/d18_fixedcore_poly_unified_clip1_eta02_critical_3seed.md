# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_fixedcore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 10.1686 +/- 4.5163 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 6.1117 +/- 1.5537 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 3.5584 +/- 0.6027 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0325 +/- 0.0006 | 0.0463 +/- 0.0006 | 0.9067 +/- 0.0051 | 0.9530 +/- 0.0061 | 320.0000 +/- 0.0000 | 36.1186 +/- 7.7014 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0057 +/- 0.0011; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0458 +/- 0.0075; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 5.3855 +/- 1.2240 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 5.4195 +/- 1.1432 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 7.6139 +/- 2.0653 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0459 +/- 0.0013 | 0.0540 +/- 0.0014 | 0.7956 +/- 0.0190 | 0.8442 +/- 0.0171 | 320.0000 +/- 0.0000 | 36.0098 +/- 5.6223 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0020 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0241 +/- 0.0081; wins/losses/ties 3/0/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 5.6331 +/- 1.3826 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 3.8832 +/- 0.8648 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 3.7935 +/- 0.5175 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0388 +/- 0.0015 | 0.0465 +/- 0.0004 | 0.8833 +/- 0.0117 | 0.9258 +/- 0.0105 | 320.0000 +/- 0.0000 | 29.9702 +/- 4.0330 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0104 +/- 0.0010; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0445 +/- 0.0057; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 3.4520 +/- 2.3532 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 2.5616 +/- 1.5375 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.4975 +/- 0.1220 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.2218 +/- 0.0609 | 0.1453 +/- 0.0223 |  |  | 320.0000 +/- 0.0000 | 23.1319 +/- 6.0698 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0499 +/- 0.0345; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 3.3267 +/- 2.2744 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 2.4173 +/- 1.0984 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 1.6827 +/- 0.1464 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.8950 +/- 0.4993 | 0.8893 +/- 0.2257 |  |  | 320.0000 +/- 0.0000 | 22.3826 +/- 2.8615 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0525 +/- 0.1452; wins/losses/ties 1/2/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.

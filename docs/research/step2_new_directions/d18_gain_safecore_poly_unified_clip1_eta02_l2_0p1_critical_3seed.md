# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 5.3306 +/- 2.8587 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 6.9168 +/- 3.9181 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 3.2682 +/- 0.2651 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0335 +/- 0.0005 | 0.0459 +/- 0.0006 | 0.8933 +/- 0.0120 | 0.9555 +/- 0.0047 | 320.0000 +/- 0.0000 | 24.6013 +/- 1.7622 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0047 +/- 0.0011; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0482 +/- 0.0057; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 6.2522 +/- 0.2796 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 6.2105 +/- 0.1652 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 6.8137 +/- 1.0077 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0454 +/- 0.0014 | 0.0534 +/- 0.0013 | 0.7800 +/- 0.0190 | 0.8281 +/- 0.0177 | 320.0000 +/- 0.0000 | 29.8101 +/- 7.5864 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0025 +/- 0.0007; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0080 +/- 0.0097; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 4.7653 +/- 2.5423 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 3.6260 +/- 1.6860 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 5.5409 +/- 3.0277 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0379 +/- 0.0013 | 0.0459 +/- 0.0002 | 0.8767 +/- 0.0100 | 0.9270 +/- 0.0091 | 320.0000 +/- 0.0000 | 20.5955 +/- 4.2886 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0113 +/- 0.0013; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0458 +/- 0.0053; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.8972 +/- 0.8193 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 2.0851 +/- 1.1181 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 2.2444 +/- 0.5132 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2380 +/- 0.0733 | 0.1520 +/- 0.0256 |  |  | 320.0000 +/- 0.0000 | 21.0982 +/- 4.9896 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0338 +/- 0.0230; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 3.7426 +/- 1.9392 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 2.9189 +/- 1.6685 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 1.8513 +/- 0.1590 |
| `d18_gain_safecore_poly_unified_0p01` | 0.8389 +/- 0.5367 | 0.8558 +/- 0.2178 |  |  | 320.0000 +/- 0.0000 | 28.6143 +/- 3.0983 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1086 +/- 0.1077; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 3}.

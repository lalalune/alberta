# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_fixedcore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 7.3709 +/- 1.1462 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 6.2425 +/- 1.0499 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 7.8046 +/- 2.7050 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0349 +/- 0.0007 | 0.0494 +/- 0.0005 | 0.8978 +/- 0.0095 | 0.9499 +/- 0.0065 | 320.0000 +/- 0.0000 | 39.7798 +/- 1.7158 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0032 +/- 0.0012; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0427 +/- 0.0077; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 5.6903 +/- 2.5783 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 5.0794 +/- 1.3152 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 5.6753 +/- 0.6770 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0495 +/- 0.0015 | 0.0581 +/- 0.0016 | 0.7722 +/- 0.0166 | 0.8151 +/- 0.0263 | 320.0000 +/- 0.0000 | 21.9893 +/- 5.0470 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0017 +/- 0.0009; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0049 +/- 0.0134; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 2.4946 +/- 0.5822 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 2.4026 +/- 0.7735 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 2.7175 +/- 0.7672 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.0423 +/- 0.0018 | 0.0505 +/- 0.0007 | 0.8678 +/- 0.0172 | 0.9221 +/- 0.0093 | 320.0000 +/- 0.0000 | 16.1979 +/- 2.8132 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0069 +/- 0.0007; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0408 +/- 0.0047; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 2.0375 +/- 1.0872 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.8847 +/- 1.1417 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.6763 +/- 0.1319 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.2652 +/- 0.0929 | 0.1656 +/- 0.0302 |  |  | 320.0000 +/- 0.0000 | 7.3477 +/- 0.5088 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0065 +/- 0.0060; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.5058 +/- 1.0081 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 2.3728 +/- 1.7878 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.6909 +/- 0.0119 |
| `d18_gain_fixedcore_poly_unified_0p01` | 0.9808 +/- 0.5389 | 0.9796 +/- 0.2510 |  |  | 320.0000 +/- 0.0000 | 9.5182 +/- 0.8947 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0333 +/- 0.1056; wins/losses/ties 1/2/0; best-D18 counts {'d18_gain_fixedcore_poly_unified_0p01': 3}.

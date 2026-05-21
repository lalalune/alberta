# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p25_basis_0p6.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 0.6038 +/- 0.0526 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 1.0536 +/- 0.1662 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 1.2535 +/- 0.2527 |
| `d18_core_0p25_basis_0p6` | 0.0364 +/- 0.0011 | 0.0906 +/- 0.0013 |  |  | 320.0000 +/- 0.0000 | 2.1089 +/- 0.0282 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1204 +/- 0.0267; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p25_basis_0p6': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.6242 +/- 0.1667 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.8203 +/- 0.1459 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 0.9781 +/- 0.0625 |
| `d18_core_0p25_basis_0p6` | 0.0510 +/- 0.0053 | 0.2334 +/- 0.0258 |  |  | 320.0000 +/- 0.0000 | 2.3986 +/- 0.2438 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3773 +/- 0.0194; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p25_basis_0p6': 3}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.7470 +/- 0.1760 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 2.8495 +/- 0.4397 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 1.9432 +/- 0.2918 |
| `d18_core_0p25_basis_0p6` | 0.0378 +/- 0.0007 | 0.0503 +/- 0.0006 | 0.8833 +/- 0.0077 | 0.9450 +/- 0.0062 | 320.0000 +/- 0.0000 | 2.8295 +/- 0.0462 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0003 +/- 0.0013; wins/losses/ties 2/1/0; best-D18 counts {'d18_core_0p25_basis_0p6': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0377 +/- 0.0073; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 1.4021 +/- 0.1481 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 1.7438 +/- 0.2809 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 1.6465 +/- 0.1424 |
| `d18_core_0p25_basis_0p6` | 0.0471 +/- 0.0014 | 0.0559 +/- 0.0012 | 0.7833 +/- 0.0201 | 0.8404 +/- 0.0119 | 320.0000 +/- 0.0000 | 3.0518 +/- 0.0574 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0007 +/- 0.0007; wins/losses/ties 2/1/0; best-D18 counts {'d18_core_0p25_basis_0p6': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0204 +/- 0.0060; wins/losses/ties 3/0/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 3.6500 +/- 1.0458 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 2.1899 +/- 0.3647 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 1.9597 +/- 0.1933 |
| `d18_core_0p25_basis_0p6` | 0.0412 +/- 0.0014 | 0.0491 +/- 0.0002 | 0.8767 +/- 0.0150 | 0.9258 +/- 0.0065 | 320.0000 +/- 0.0000 | 2.9187 +/- 0.1337 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0080 +/- 0.0011; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p25_basis_0p6': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0445 +/- 0.0060; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.6150 +/- 0.4767 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.5067 +/- 0.2457 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.3943 +/- 0.3564 |
| `d18_core_0p25_basis_0p6` | 0.2277 +/- 0.0676 | 0.1478 +/- 0.0237 |  |  | 320.0000 +/- 0.0000 | 2.2433 +/- 0.1043 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0440 +/- 0.0278; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p25_basis_0p6': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 1.2713 +/- 0.4422 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 1.0229 +/- 0.4634 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.9130 +/- 0.0759 |
| `d18_core_0p25_basis_0p6` | 0.8410 +/- 0.1205 | 0.9428 +/- 0.2047 |  |  | 320.0000 +/- 0.0000 | 6.9211 +/- 2.3762 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3074 +/- 0.1497; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p25_basis_0p6': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 2.8738 +/- 1.0914 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 3.1706 +/- 0.8484 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 3.2035 +/- 0.9123 |
| `d18_core_0p25_basis_0p6` | 1.0380 +/- 0.6272 | 1.1072 +/- 0.2764 |  |  | 320.0000 +/- 0.0000 | 12.3769 +/- 1.6956 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0906 +/- 0.0179; wins/losses/ties 0/3/0; best-D18 counts {'d18_core_0p25_basis_0p6': 3}.

# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3952 +/- 0.0203 | 0.5573 +/- 0.0111 |  |  |  | 4.2337 +/- 0.7524 |
| `mlp_h128` | 0.4233 +/- 0.0144 | 0.5731 +/- 0.0089 |  |  |  | 4.5468 +/- 1.2396 |
| `mlp_h64_64` | 0.1811 +/- 0.0157 | 0.3884 +/- 0.0142 |  |  |  | 5.4232 +/- 1.7918 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0373 +/- 0.0006 | 0.0871 +/- 0.0007 |  |  | 320.0000 +/- 0.0000 | 47.5334 +/- 8.6974 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1438 +/- 0.0158; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4929 +/- 0.0342 | 0.6412 +/- 0.0189 |  |  |  | 1.2847 +/- 0.1328 |
| `mlp_h128` | 0.5657 +/- 0.0339 | 0.6617 +/- 0.0220 |  |  |  | 1.1266 +/- 0.2053 |
| `mlp_h64_64` | 0.6451 +/- 0.0394 | 0.7386 +/- 0.0285 |  |  |  | 1.5841 +/- 0.1716 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0407 +/- 0.0044 | 0.1078 +/- 0.0109 |  |  | 320.0000 +/- 0.0000 | 17.2697 +/- 1.3969 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4522 +/- 0.0303; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 1.9067 +/- 0.2847 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 1.5655 +/- 0.3688 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 1.4941 +/- 0.2342 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0351 +/- 0.0006 | 0.0468 +/- 0.0004 | 0.8687 +/- 0.0051 | 0.9458 +/- 0.0043 | 320.0000 +/- 0.0000 | 9.5049 +/- 0.6373 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0044 +/- 0.0006; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0458 +/- 0.0061; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 1.4751 +/- 0.1763 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 1.2864 +/- 0.1895 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 1.7878 +/- 0.3217 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0422 +/- 0.0015 | 0.0517 +/- 0.0009 | 0.8017 +/- 0.0146 | 0.8440 +/- 0.0126 | 320.0000 +/- 0.0000 | 13.4382 +/- 1.6006 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0058 +/- 0.0009; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0080 +/- 0.0095; wins/losses/ties 6/4/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 1.2802 +/- 0.2461 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 1.3051 +/- 0.2237 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 1.5500 +/- 0.3112 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0377 +/- 0.0005 | 0.0459 +/- 0.0002 | 0.8733 +/- 0.0041 | 0.9223 +/- 0.0046 | 320.0000 +/- 0.0000 | 8.3120 +/- 1.3169 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0107 +/- 0.0004; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0442 +/- 0.0058; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.5137 +/- 0.1302 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.4489 +/- 0.1488 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.4979 +/- 0.0833 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2244 +/- 0.0285 | 0.1764 +/- 0.0139 |  |  | 320.0000 +/- 0.0000 | 3.6123 +/- 0.2905 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0449 +/- 0.0187; wins/losses/ties 9/1/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.6271 +/- 0.3675 | 1.4277 +/- 0.1462 |  |  |  | 0.3052 +/- 0.0332 |
| `mlp_h128` | 1.6349 +/- 0.3642 | 1.4399 +/- 0.1466 |  |  |  | 0.3751 +/- 0.0358 |
| `mlp_h64_64` | 1.6000 +/- 0.3571 | 1.4221 +/- 0.1407 |  |  |  | 0.4420 +/- 0.0383 |
| `d18_gain_safecore_poly_unified_0p01` | 1.0595 +/- 0.2397 | 0.8799 +/- 0.0942 |  |  | 320.0000 +/- 0.0000 | 3.4417 +/- 0.4083 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5345 +/- 0.1240; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.2157 | 1.0156 +/- 0.0967 |  |  |  | 0.4257 +/- 0.0782 |
| `mlp_h128` | 1.0368 +/- 0.2152 | 1.0187 +/- 0.0969 |  |  |  | 0.5286 +/- 0.1270 |
| `mlp_h64_64` | 1.0051 +/- 0.2125 | 0.9966 +/- 0.0942 |  |  |  | 0.5491 +/- 0.0950 |
| `d18_gain_safecore_poly_unified_0p01` | 0.8565 +/- 0.1842 | 0.7658 +/- 0.0807 |  |  | 320.0000 +/- 0.0000 | 4.4465 +/- 0.5834 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1485 +/- 0.0353; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.

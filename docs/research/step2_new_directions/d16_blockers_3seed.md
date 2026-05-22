# D16 Additive Universal Learner Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: additive_group_memory, additive_group_memory_poly.

Each candidate is one additive predictor updated from one global residual at every step. Positive candidate-vs-MLP differences favor the additive learner.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  |  | 0.3057 +/- 0.0652 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  |  | 0.3182 +/- 0.0556 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  |  | 0.4030 +/- 0.0454 |
| `additive_group_memory` | 7.4337 +/- 2.5735 | 3.2354 +/- 0.9205 |  |  | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 1.0622 +/- 0.0650 |
| `additive_group_memory_poly` | 2217.7507 +/- 770.8844 | 711.2115 +/- 262.7712 |  |  | 128.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | 1.9135 +/- 0.3123 |

`final_window_mse` best-additive-vs-best-MLP diff: -7.2768 +/- 2.5566; wins/losses/ties 0/3/0; best-additive counts {'additive_group_memory': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  |  | 0.2460 +/- 0.0013 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  |  | 0.3564 +/- 0.0616 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  |  | 0.5354 +/- 0.0713 |
| `additive_group_memory` | 12.4617 +/- 0.6566 | 5.8377 +/- 0.5037 |  |  | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 1.6384 +/- 0.0531 |
| `additive_group_memory_poly` | 3299.4597 +/- 1923.7522 | 1013.6316 +/- 545.1335 |  |  | 128.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | 2.1289 +/- 0.1802 |

`final_window_mse` best-additive-vs-best-MLP diff: -12.0335 +/- 0.6718; wins/losses/ties 0/3/0; best-additive counts {'additive_group_memory': 3}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  |  | 1.3992 +/- 0.2593 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  |  | 1.3522 +/- 0.0780 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  |  | 0.9863 +/- 0.1147 |
| `additive_group_memory` | 0.5137 +/- 0.0066 | 0.2763 +/- 0.0087 | 0.3311 +/- 0.0113 | 0.4143 +/- 0.0182 | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 4.0577 +/- 0.9656 |
| `additive_group_memory_poly` | 32.3021 +/- 0.7507 | 17.1479 +/- 0.1266 | 0.1278 +/- 0.0022 | 0.0823 +/- 0.0034 | 128.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | 4.5041 +/- 0.3347 |

`final_window_mse` best-additive-vs-best-MLP diff: -0.4755 +/- 0.0058; wins/losses/ties 0/3/0; best-additive counts {'additive_group_memory': 3}.
`test_accuracy` best-additive-vs-best-MLP diff: -0.4929 +/- 0.0167; wins/losses/ties 0/3/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  |  | 0.6886 +/- 0.0361 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  |  | 0.7487 +/- 0.0375 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  |  | 1.0178 +/- 0.1802 |
| `additive_group_memory` | 0.9028 +/- 0.1159 | 0.4916 +/- 0.0392 | 0.2422 +/- 0.0195 | 0.2511 +/- 0.0108 | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3.4772 +/- 0.5583 |
| `additive_group_memory_poly` | 38.1693 +/- 0.5278 | 22.2522 +/- 0.2588 | 0.1167 +/- 0.0069 | 0.0983 +/- 0.0348 | 128.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | 4.0486 +/- 0.5074 |

`final_window_mse` best-additive-vs-best-MLP diff: -0.8550 +/- 0.1157; wins/losses/ties 0/3/0; best-additive counts {'additive_group_memory': 3}.
`test_accuracy` best-additive-vs-best-MLP diff: -0.5690 +/- 0.0051; wins/losses/ties 0/3/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  |  | 0.7065 +/- 0.0785 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  |  | 0.9658 +/- 0.2822 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  |  | 1.0452 +/- 0.2364 |
| `additive_group_memory` | 0.4431 +/- 0.0305 | 0.2402 +/- 0.0045 | 0.3433 +/- 0.0133 | 0.4521 +/- 0.0213 | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3.0249 +/- 0.2938 |
| `additive_group_memory_poly` | 34.9666 +/- 3.9078 | 17.5805 +/- 0.9187 | 0.1200 +/- 0.0038 | 0.1224 +/- 0.0166 | 128.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | 5.1796 +/- 0.1420 |

`final_window_mse` best-additive-vs-best-MLP diff: -0.3939 +/- 0.0287; wins/losses/ties 0/3/0; best-additive counts {'additive_group_memory': 3}.
`test_accuracy` best-additive-vs-best-MLP diff: -0.4292 +/- 0.0184; wins/losses/ties 0/3/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  |  | 0.6602 +/- 0.4064 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  |  | 0.4890 +/- 0.2397 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  |  | 0.4587 +/- 0.0785 |
| `additive_group_memory` | 3.5701 +/- 0.8621 | 1.2680 +/- 0.2508 |  |  | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 1.9661 +/- 0.5054 |
| `additive_group_memory_poly` | 2187.9331 +/- 562.2113 | 638.1518 +/- 166.7823 |  |  | 128.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | 2.8174 +/- 0.4877 |

`final_window_mse` best-additive-vs-best-MLP diff: -3.2984 +/- 0.7667; wins/losses/ties 0/3/0; best-additive counts {'additive_group_memory': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  |  | 0.4975 +/- 0.1973 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  |  | 0.4729 +/- 0.1804 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  |  | 0.3899 +/- 0.0691 |
| `additive_group_memory` | 5.7550 +/- 0.3420 | 3.7688 +/- 0.3959 |  |  | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 1.1578 +/- 0.1872 |
| `additive_group_memory_poly` | 7357.1059 +/- 3103.0158 | 2618.9351 +/- 1211.0044 |  |  | 128.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | 2.1374 +/- 0.3144 |

`final_window_mse` best-additive-vs-best-MLP diff: -4.6065 +/- 0.1836; wins/losses/ties 0/3/0; best-additive counts {'additive_group_memory': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  |  | 1.0229 +/- 0.6774 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  |  | 1.3044 +/- 1.0086 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  |  | 0.7788 +/- 0.2103 |
| `additive_group_memory` | 7.6501 +/- 1.1494 | 4.5435 +/- 0.5727 |  |  | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 2.1734 +/- 0.5223 |
| `additive_group_memory_poly` | 1526.5450 +/- 473.3369 | 443.9897 +/- 140.8377 |  |  | 128.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | 3.2954 +/- 0.2398 |

`final_window_mse` best-additive-vs-best-MLP diff: -6.7027 +/- 0.5400; wins/losses/ties 0/3/0; best-additive counts {'additive_group_memory': 3}.

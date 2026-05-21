# D15 Groupwise Basis LMS Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: groupwise_no_poly.

Each candidate is one additive predictor with blockwise normalized LMS. Every included basis family updates from the same residual at every step; there is no prediction router or expert selection.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 0.9590 +/- 0.6195 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 0.8723 +/- 0.4630 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 0.6824 +/- 0.2812 |
| `groupwise_no_poly` | 0.0443 +/- 0.0011 | 0.0780 +/- 0.0010 |  |  | 294.0000 +/- 0.0000 | 0.1817 +/- 0.0433 |

`final_window_mse` best-groupwise-vs-best-MLP diff: +0.1125 +/- 0.0274; wins/losses/ties 3/0/0; best-groupwise counts {'groupwise_no_poly': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.5186 +/- 0.0677 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.4643 +/- 0.0302 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 0.5260 +/- 0.0361 |
| `groupwise_no_poly` | 0.5413 +/- 0.0806 | 0.9330 +/- 0.0895 |  |  | 294.0000 +/- 0.0000 | 0.1279 +/- 0.0333 |

`final_window_mse` best-groupwise-vs-best-MLP diff: -0.1131 +/- 0.0564; wins/losses/ties 0/3/0; best-groupwise counts {'groupwise_no_poly': 3}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8611 +/- 0.1893 | 0.9917 +/- 0.0835 |  |  |  | 0.3878 +/- 0.0158 |
| `mlp_h128` | 0.9468 +/- 0.2304 | 1.0382 +/- 0.0979 |  |  |  | 0.3683 +/- 0.0449 |
| `mlp_h64_64` | 0.9490 +/- 0.2284 | 1.0421 +/- 0.1050 |  |  |  | 0.4737 +/- 0.0342 |
| `groupwise_no_poly` | 0.5391 +/- 0.1587 | 0.7349 +/- 0.0744 |  |  | 294.0000 +/- 0.0000 | 0.1234 +/- 0.0128 |

`final_window_mse` best-groupwise-vs-best-MLP diff: +0.3209 +/- 0.0376; wins/losses/ties 3/0/0; best-groupwise counts {'groupwise_no_poly': 3}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8637 +/- 0.0489 | 1.0904 +/- 0.0786 |  |  |  | 0.3875 +/- 0.0211 |
| `mlp_h128` | 0.8693 +/- 0.0485 | 1.0951 +/- 0.0761 |  |  |  | 0.5740 +/- 0.1044 |
| `mlp_h64_64` | 0.6059 +/- 0.0368 | 0.9161 +/- 0.0726 |  |  |  | 0.7933 +/- 0.1356 |
| `groupwise_no_poly` | 0.6126 +/- 0.0378 | 1.0400 +/- 0.0883 |  |  | 294.0000 +/- 0.0000 | 0.2721 +/- 0.0944 |

`final_window_mse` best-groupwise-vs-best-MLP diff: -0.0067 +/- 0.0201; wins/losses/ties 1/2/0; best-groupwise counts {'groupwise_no_poly': 3}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.1335 +/- 0.1768 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 1.1296 +/- 0.3021 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 0.9637 +/- 0.1542 |
| `groupwise_no_poly` | 0.0540 +/- 0.0001 | 0.0626 +/- 0.0005 | 0.7856 +/- 0.0095 | 0.8448 +/- 0.0097 | 330.0000 +/- 0.0000 | 0.4185 +/- 0.0760 |

`final_window_mse` best-groupwise-vs-best-MLP diff: -0.0159 +/- 0.0009; wins/losses/ties 0/3/0; best-groupwise counts {'groupwise_no_poly': 3}.
`test_accuracy` best-groupwise-vs-best-MLP diff: -0.0625 +/- 0.0101; wins/losses/ties 0/3/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.9342 +/- 0.2110 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 0.8461 +/- 0.1145 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.8630 +/- 0.0332 |
| `groupwise_no_poly` | 0.0632 +/- 0.0003 | 0.0683 +/- 0.0012 | 0.6900 +/- 0.0115 | 0.7669 +/- 0.0161 | 330.0000 +/- 0.0000 | 0.2894 +/- 0.1046 |

`final_window_mse` best-groupwise-vs-best-MLP diff: -0.0154 +/- 0.0007; wins/losses/ties 0/3/0; best-groupwise counts {'groupwise_no_poly': 3}.
`test_accuracy` best-groupwise-vs-best-MLP diff: -0.0532 +/- 0.0083; wins/losses/ties 0/3/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 0.9189 +/- 0.0899 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 1.0202 +/- 0.1507 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 0.9248 +/- 0.1565 |
| `groupwise_no_poly` | 0.0540 +/- 0.0015 | 0.0599 +/- 0.0004 | 0.8133 +/- 0.0168 | 0.8534 +/- 0.0140 | 330.0000 +/- 0.0000 | 0.2281 +/- 0.0795 |

`final_window_mse` best-groupwise-vs-best-MLP diff: -0.0048 +/- 0.0016; wins/losses/ties 0/3/0; best-groupwise counts {'groupwise_no_poly': 3}.
`test_accuracy` best-groupwise-vs-best-MLP diff: -0.0278 +/- 0.0235; wins/losses/ties 1/2/0.

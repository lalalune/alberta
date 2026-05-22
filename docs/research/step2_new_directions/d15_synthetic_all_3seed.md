# D15 Groupwise Basis LMS Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: groupwise_canonical, groupwise_no_poly, groupwise_slow_poly, groupwise_tanh_fourier_fast.

Each candidate is one additive predictor with blockwise normalized LMS. Every included basis family updates from the same residual at every step; there is no prediction router or expert selection.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.6521 +/- 0.1543 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.7042 +/- 0.2787 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.5296 +/- 0.1185 |
| `groupwise_canonical` | 29142.6404 +/- 10211.2715 | 7428.2696 +/- 2558.7203 |  |  | 396.0000 +/- 0.0000 | 0.1515 +/- 0.0550 |
| `groupwise_no_poly` | 0.2383 +/- 0.0745 | 0.1550 +/- 0.0260 |  |  | 312.0000 +/- 0.0000 | 0.2353 +/- 0.1016 |
| `groupwise_slow_poly` | 2.0758 +/- 0.4601 | 0.7649 +/- 0.0964 |  |  | 396.0000 +/- 0.0000 | 0.1437 +/- 0.0355 |
| `groupwise_tanh_fourier_fast` | 251.8412 +/- 27.9842 | 69.2497 +/- 8.4711 |  |  | 396.0000 +/- 0.0000 | 0.2115 +/- 0.0614 |

`final_window_mse` best-groupwise-vs-best-MLP diff: +0.0334 +/- 0.0209; wins/losses/ties 3/0/0; best-groupwise counts {'groupwise_no_poly': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.6058 +/- 0.1536 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.5612 +/- 0.1596 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.6183 +/- 0.1195 |
| `groupwise_canonical` | 166275561287.1241 +/- 129480702915.2874 | 41607756921.1973 +/- 32392109072.1605 |  |  | 329.0000 +/- 0.0000 | 0.2632 +/- 0.0394 |
| `groupwise_no_poly` | 0.7943 +/- 0.0955 | 0.9100 +/- 0.2047 |  |  | 294.0000 +/- 0.0000 | 0.3040 +/- 0.0547 |
| `groupwise_slow_poly` | 1020.1138 +/- 587.3454 | 282.1897 +/- 161.3797 |  |  | 329.0000 +/- 0.0000 | 0.2791 +/- 0.0672 |
| `groupwise_tanh_fourier_fast` | 20920670.7428 +/- 10728684.6584 | 5277273.2644 +/- 2703582.3477 |  |  | 329.0000 +/- 0.0000 | 0.1627 +/- 0.0382 |

`final_window_mse` best-groupwise-vs-best-MLP diff: +0.3542 +/- 0.1818; wins/losses/ties 3/0/0; best-groupwise counts {'groupwise_no_poly': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.2671 +/- 0.8541 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.0850 +/- 0.8014 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.6856 +/- 0.2719 |
| `groupwise_canonical` | 2430.3646 +/- 796.6928 | 633.3406 +/- 202.1733 |  |  | 495.0000 +/- 0.0000 | 0.1410 +/- 0.0073 |
| `groupwise_no_poly` | 1.2110 +/- 0.7848 | 1.2895 +/- 0.3309 |  |  | 330.0000 +/- 0.0000 | 0.1922 +/- 0.0257 |
| `groupwise_slow_poly` | 1.4777 +/- 0.5628 | 1.3693 +/- 0.3503 |  |  | 495.0000 +/- 0.0000 | 0.2634 +/- 0.0502 |
| `groupwise_tanh_fourier_fast` | 24.8615 +/- 6.3975 | 8.9011 +/- 2.6138 |  |  | 495.0000 +/- 0.0000 | 0.2459 +/- 0.0902 |

`final_window_mse` best-groupwise-vs-best-MLP diff: -0.2004 +/- 0.0777; wins/losses/ties 0/3/0; best-groupwise counts {'groupwise_no_poly': 2, 'groupwise_slow_poly': 1}.

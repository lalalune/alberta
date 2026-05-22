# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.2760 +/- 0.0925 | 0.2002 +/- 0.0304 |  |  |
| `recursive_features` | 0.9415 +/- 0.3487 | 0.6405 +/- 0.1299 |  |  |
| `polynomial_features` | 1.5724 +/- 0.6786 | 0.9813 +/- 0.1784 |  |  |
| `fourier_features` | 0.4132 +/- 0.1248 | 0.2622 +/- 0.0415 |  |  |
| `tanh_random_features` | 0.2843 +/- 0.0877 | 0.1901 +/- 0.0285 |  |  |
| `mlp_32x32_s01_no_ln` | 0.4810 +/- 0.2083 | 0.3090 +/- 0.0690 |  |  |
| `mlp_64x64_s01_no_ln` | 0.5581 +/- 0.2385 | 0.3283 +/- 0.0714 |  |  |
| `mlp_32x32` | 0.4908 +/- 0.2024 | 0.3256 +/- 0.0680 |  |  |
| `mlp_h64` | 0.2758 +/- 0.0955 | 0.1798 +/- 0.0286 |  |  |
| `mlp_h128` | 0.2757 +/- 0.0919 | 0.1799 +/- 0.0299 |  |  |
| `mlp_h64_64` | 0.3683 +/- 0.1354 | 0.2323 +/- 0.0404 |  |  |
| `upgd_low_noise` | 0.4463 +/- 0.1682 | 0.2716 +/- 0.0554 |  |  |
| `dynamic_sparse` | 0.4086 +/- 0.1523 | 0.2563 +/- 0.0493 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0033 +/- 0.0036; wins/losses/ties 1/2/0.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.7724 +/- 0.5041 | 0.8884 +/- 0.2146 |  |  |
| `recursive_features` | 0.9570 +/- 0.6651 | 0.9717 +/- 0.2227 |  |  |
| `polynomial_features` | 0.8507 +/- 0.3960 | 0.8710 +/- 0.2091 |  |  |
| `fourier_features` | 1.0094 +/- 0.6904 | 1.0752 +/- 0.2645 |  |  |
| `tanh_random_features` | 0.9474 +/- 0.6699 | 1.0066 +/- 0.2414 |  |  |
| `mlp_32x32_s01_no_ln` | 0.9170 +/- 0.6354 | 0.9781 +/- 0.2300 |  |  |
| `mlp_64x64_s01_no_ln` | 0.9064 +/- 0.6316 | 0.9593 +/- 0.2281 |  |  |
| `mlp_32x32` | 0.9142 +/- 0.6476 | 0.9711 +/- 0.2329 |  |  |
| `mlp_h64` | 0.9662 +/- 0.6492 | 1.0431 +/- 0.2395 |  |  |
| `mlp_h128` | 0.9714 +/- 0.6512 | 1.0442 +/- 0.2380 |  |  |
| `mlp_h64_64` | 0.9456 +/- 0.6441 | 1.0231 +/- 0.2348 |  |  |
| `upgd_low_noise` | 0.9184 +/- 0.6359 | 0.9763 +/- 0.2242 |  |  |
| `dynamic_sparse` | 0.9175 +/- 0.6354 | 0.9826 +/- 0.2303 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.1285 +/- 0.1303; wins/losses/ties 1/2/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/2 configured datasets.

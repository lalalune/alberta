# Step 2 Conclusive Learner Candidate

Protocol: 10 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.2168 +/- 0.0311 | 0.2143 +/- 0.0180 |  |  |
| `recursive_features` | 1.1168 +/- 0.1822 | 0.8638 +/- 0.0953 |  |  |
| `polynomial_features` | 1.5728 +/- 0.2411 | 1.1903 +/- 0.1006 |  |  |
| `fourier_features` | 0.4210 +/- 0.0637 | 0.3282 +/- 0.0300 |  |  |
| `tanh_random_features` | 0.2199 +/- 0.0306 | 0.1773 +/- 0.0143 |  |  |
| `mlp_32x32_s01_no_ln` | 0.4610 +/- 0.0796 | 0.3736 +/- 0.0378 |  |  |
| `mlp_64x64_s01_no_ln` | 0.5195 +/- 0.0917 | 0.4071 +/- 0.0406 |  |  |
| `mlp_32x32` | 0.4714 +/- 0.0778 | 0.3933 +/- 0.0355 |  |  |
| `mlp_h64` | 0.2716 +/- 0.0436 | 0.2173 +/- 0.0195 |  |  |
| `mlp_h128` | 0.2705 +/- 0.0448 | 0.2155 +/- 0.0193 |  |  |
| `mlp_h64_64` | 0.3351 +/- 0.0549 | 0.2678 +/- 0.0223 |  |  |
| `upgd_low_noise` | 0.5154 +/- 0.1205 | 0.3718 +/- 0.0506 |  |  |
| `dynamic_sparse` | 0.4634 +/- 0.1045 | 0.3483 +/- 0.0456 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0509 +/- 0.0133; wins/losses/ties 10/0/0.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.8912 +/- 0.1801 | 0.8719 +/- 0.0850 |  |  |
| `recursive_features` | 1.0007 +/- 0.2095 | 0.9312 +/- 0.0898 |  |  |
| `polynomial_features` | 0.9786 +/- 0.1701 | 0.8834 +/- 0.0848 |  |  |
| `fourier_features` | 1.0633 +/- 0.2284 | 1.0355 +/- 0.1039 |  |  |
| `tanh_random_features` | 1.0962 +/- 0.2397 | 1.0627 +/- 0.1051 |  |  |
| `mlp_32x32_s01_no_ln` | 0.9632 +/- 0.2093 | 0.9431 +/- 0.0928 |  |  |
| `mlp_64x64_s01_no_ln` | 0.9542 +/- 0.2084 | 0.9304 +/- 0.0921 |  |  |
| `mlp_32x32` | 0.9544 +/- 0.2109 | 0.9313 +/- 0.0932 |  |  |
| `mlp_h64` | 1.0298 +/- 0.2147 | 1.0163 +/- 0.0962 |  |  |
| `mlp_h128` | 1.0377 +/- 0.2152 | 1.0189 +/- 0.0963 |  |  |
| `mlp_h64_64` | 1.0006 +/- 0.2120 | 0.9952 +/- 0.0950 |  |  |
| `upgd_low_noise` | 0.9652 +/- 0.2078 | 0.9457 +/- 0.0911 |  |  |
| `dynamic_sparse` | 0.9650 +/- 0.2080 | 0.9530 +/- 0.0923 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0554 +/- 0.0447; wins/losses/ties 6/4/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 2/2 configured datasets.

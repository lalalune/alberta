# Step 2 Conclusive Learner Candidate

Protocol: 10 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0437 +/- 0.0080 | 0.0786 +/- 0.0065 |  |  |
| `recursive_features` | 0.1146 +/- 0.0126 | 0.1483 +/- 0.0076 |  |  |
| `polynomial_features` | 0.0408 +/- 0.0067 | 0.0864 +/- 0.0051 |  |  |
| `fourier_features` | 0.0727 +/- 0.0108 | 0.0823 +/- 0.0081 |  |  |
| `tanh_random_features` | 0.0787 +/- 0.0110 | 0.0821 +/- 0.0083 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0667 +/- 0.0094 | 0.0801 +/- 0.0075 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0652 +/- 0.0097 | 0.0760 +/- 0.0077 |  |  |
| `mlp_32x32` | 0.0668 +/- 0.0099 | 0.0871 +/- 0.0077 |  |  |
| `mlp_h64` | 0.0955 +/- 0.0115 | 0.1123 +/- 0.0076 |  |  |
| `mlp_h128` | 0.0971 +/- 0.0108 | 0.1124 +/- 0.0075 |  |  |
| `mlp_h64_64` | 0.1132 +/- 0.0109 | 0.1312 +/- 0.0076 |  |  |
| `upgd_low_noise` | 0.1017 +/- 0.0115 | 0.1190 +/- 0.0082 |  |  |
| `dynamic_sparse` | 0.0994 +/- 0.0106 | 0.1187 +/- 0.0075 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0205 +/- 0.0049; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.2238 +/- 0.0332 | 0.2184 +/- 0.0191 |  |  |
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

`final_window_mse` conclusive-vs-best-MLP diff: +0.0439 +/- 0.0112; wins/losses/ties 10/0/0.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.6822 +/- 0.1188 | 0.7668 +/- 0.0823 |  |  |
| `recursive_features` | 2.7647 +/- 0.3489 | 2.4509 +/- 0.1479 |  |  |
| `polynomial_features` | 4.7805 +/- 1.1346 | 3.9505 +/- 0.4488 |  |  |
| `fourier_features` | 0.6678 +/- 0.1200 | 0.5623 +/- 0.0525 |  |  |
| `tanh_random_features` | 1.6799 +/- 0.3898 | 1.5032 +/- 0.1571 |  |  |
| `mlp_32x32_s01_no_ln` | 1.5747 +/- 0.3707 | 1.3592 +/- 0.1387 |  |  |
| `mlp_64x64_s01_no_ln` | 1.5518 +/- 0.3626 | 1.3501 +/- 0.1361 |  |  |
| `mlp_32x32` | 1.5293 +/- 0.3562 | 1.3142 +/- 0.1325 |  |  |
| `mlp_h64` | 1.6150 +/- 0.3652 | 1.4323 +/- 0.1482 |  |  |
| `mlp_h128` | 1.6315 +/- 0.3686 | 1.4393 +/- 0.1498 |  |  |
| `mlp_h64_64` | 1.5947 +/- 0.3622 | 1.4223 +/- 0.1466 |  |  |
| `upgd_low_noise` | 1.6402 +/- 0.3579 | 1.4166 +/- 0.1416 |  |  |
| `dynamic_sparse` | 1.6223 +/- 0.3641 | 1.4074 +/- 0.1421 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.8120 +/- 0.2464; wins/losses/ties 9/1/0.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.8851 +/- 0.1792 | 0.8645 +/- 0.0850 |  |  |
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

`final_window_mse` conclusive-vs-best-MLP diff: +0.0616 +/- 0.0403; wins/losses/ties 6/4/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 4/4 configured datasets.

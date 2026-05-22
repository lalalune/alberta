# Step 2 Conclusive Learner Candidate

Protocol: 10 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.9259 +/- 0.1924 | 0.8948 +/- 0.0900 |  |  |
| `recursive_features` | 1.0007 +/- 0.2095 | 0.9312 +/- 0.0898 |  |  |
| `polynomial_features` | 1.0498 +/- 0.1731 | 0.9244 +/- 0.0881 |  |  |
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

`final_window_mse` conclusive-vs-best-MLP diff: +0.0208 +/- 0.0378; wins/losses/ties 5/5/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

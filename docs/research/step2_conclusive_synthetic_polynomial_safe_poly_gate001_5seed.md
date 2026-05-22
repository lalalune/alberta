# Step 2 Conclusive Learner Candidate

Protocol: 5 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.8819 +/- 0.3207 | 0.8546 +/- 0.1768 |  |  |
| `recursive_features` | 1.0044 +/- 0.3924 | 0.9374 +/- 0.1848 |  |  |
| `polynomial_features` | 0.9765 +/- 0.2998 | 0.8573 +/- 0.1738 |  |  |
| `fourier_features` | 1.0935 +/- 0.4248 | 1.0416 +/- 0.2157 |  |  |
| `tanh_random_features` | 1.1315 +/- 0.4480 | 1.0622 +/- 0.2180 |  |  |
| `mlp_32x32_s01_no_ln` | 0.9956 +/- 0.3905 | 0.9479 +/- 0.1927 |  |  |
| `mlp_64x64_s01_no_ln` | 0.9853 +/- 0.3887 | 0.9324 +/- 0.1912 |  |  |
| `mlp_32x32` | 0.9804 +/- 0.3941 | 0.9373 +/- 0.1936 |  |  |
| `mlp_h64` | 1.0605 +/- 0.4004 | 1.0148 +/- 0.1996 |  |  |
| `mlp_h128` | 1.0628 +/- 0.4004 | 1.0169 +/- 0.1997 |  |  |
| `mlp_h64_64` | 1.0301 +/- 0.3946 | 0.9940 +/- 0.1971 |  |  |
| `upgd_low_noise` | 0.9920 +/- 0.3881 | 0.9480 +/- 0.1890 |  |  |
| `dynamic_sparse` | 0.9925 +/- 0.3879 | 0.9539 +/- 0.1920 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0905 +/- 0.0981; wins/losses/ties 2/3/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

# Step 2 Conclusive Learner Candidate

Protocol: 10 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.2125 +/- 0.0301 | 0.2126 +/- 0.0173 |  |  |
| `recursive_features` | 1.1168 +/- 0.1822 | 0.8638 +/- 0.0953 |  |  |
| `polynomial_features` | 1.5728 +/- 0.2411 | 1.1903 +/- 0.1006 |  |  |
| `fourier_features` | 0.4210 +/- 0.0637 | 0.3282 +/- 0.0300 |  |  |
| `tanh_random_features` | 0.2206 +/- 0.0305 | 0.1790 +/- 0.0137 |  |  |
| `mlp_32x32_s01_no_ln` | 0.4610 +/- 0.0796 | 0.3736 +/- 0.0378 |  |  |
| `mlp_64x64_s01_no_ln` | 0.5195 +/- 0.0917 | 0.4071 +/- 0.0406 |  |  |
| `mlp_32x32` | 0.4714 +/- 0.0778 | 0.3933 +/- 0.0355 |  |  |
| `mlp_h64` | 0.2716 +/- 0.0436 | 0.2173 +/- 0.0195 |  |  |
| `mlp_h128` | 0.2705 +/- 0.0448 | 0.2155 +/- 0.0193 |  |  |
| `mlp_h64_64` | 0.3351 +/- 0.0549 | 0.2678 +/- 0.0223 |  |  |
| `upgd_low_noise` | 0.5154 +/- 0.1205 | 0.3718 +/- 0.0506 |  |  |
| `dynamic_sparse` | 0.4634 +/- 0.1045 | 0.3483 +/- 0.0456 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0551 +/- 0.0159; wins/losses/ties 10/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

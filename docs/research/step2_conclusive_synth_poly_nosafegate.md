# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.7667 +/- 0.4979 | 0.8792 +/- 0.2088 |  |  |
| `recursive_features` | 0.9570 +/- 0.6651 | 0.9717 +/- 0.2227 |  |  |
| `polynomial_features` | 0.8507 +/- 0.3960 | 0.8710 +/- 0.2091 |  |  |
| `fourier_features` | 1.0094 +/- 0.6904 | 1.0752 +/- 0.2645 |  |  |
| `tanh_random_features` | 1.0354 +/- 0.7231 | 1.0927 +/- 0.2605 |  |  |
| `mlp_32x32_s01_no_ln` | 0.9170 +/- 0.6354 | 0.9781 +/- 0.2300 |  |  |
| `mlp_64x64_s01_no_ln` | 0.9064 +/- 0.6316 | 0.9593 +/- 0.2281 |  |  |
| `mlp_32x32` | 0.9142 +/- 0.6476 | 0.9711 +/- 0.2329 |  |  |
| `mlp_h64` | 0.9662 +/- 0.6492 | 1.0431 +/- 0.2395 |  |  |
| `mlp_h128` | 0.9714 +/- 0.6512 | 1.0442 +/- 0.2380 |  |  |
| `mlp_h64_64` | 0.9456 +/- 0.6441 | 1.0231 +/- 0.2348 |  |  |
| `upgd_low_noise` | 0.9184 +/- 0.6359 | 0.9763 +/- 0.2242 |  |  |
| `dynamic_sparse` | 0.9175 +/- 0.6354 | 0.9826 +/- 0.2303 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.1342 +/- 0.1365; wins/losses/ties 1/2/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.2259 +/- 0.0705 | 0.1768 +/- 0.0245 |  |  |
| `recursive_features` | 0.9415 +/- 0.3487 | 0.6405 +/- 0.1299 |  |  |
| `polynomial_features` | 1.5724 +/- 0.6786 | 0.9813 +/- 0.1784 |  |  |
| `fourier_features` | 0.4132 +/- 0.1248 | 0.2622 +/- 0.0415 |  |  |
| `tanh_random_features` | 0.2259 +/- 0.0705 | 0.1502 +/- 0.0232 |  |  |
| `mlp_32x32_s01_no_ln` | 0.4810 +/- 0.2083 | 0.3090 +/- 0.0690 |  |  |
| `mlp_64x64_s01_no_ln` | 0.5581 +/- 0.2385 | 0.3283 +/- 0.0714 |  |  |
| `mlp_32x32` | 0.4908 +/- 0.2024 | 0.3256 +/- 0.0680 |  |  |
| `mlp_h64` | 0.2758 +/- 0.0955 | 0.1798 +/- 0.0286 |  |  |
| `mlp_h128` | 0.2757 +/- 0.0919 | 0.1799 +/- 0.0299 |  |  |
| `mlp_h64_64` | 0.3683 +/- 0.1354 | 0.2323 +/- 0.0404 |  |  |
| `upgd_low_noise` | 0.4463 +/- 0.1682 | 0.2716 +/- 0.0554 |  |  |
| `dynamic_sparse` | 0.4086 +/- 0.1523 | 0.2563 +/- 0.0493 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0467 +/- 0.0230; wins/losses/ties 3/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

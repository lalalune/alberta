# Step 2 Conclusive Learner Candidate

Protocol: 1 seeds, 300 steps, final window 100; route loss decay=0.99, warmup=50, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.2197 +/- 0.0000 | 0.3789 +/- 0.0000 |  |  |
| `recursive_features` | 0.2340 +/- 0.0000 | 0.3304 +/- 0.0000 |  |  |
| `polynomial_features` | 0.2283 +/- 0.0000 | 0.4550 +/- 0.0000 |  |  |
| `mlp_32x32_s01_no_ln` | 0.6819 +/- 0.0000 | 0.8596 +/- 0.0000 |  |  |
| `mlp_64x64_s01_no_ln` | 0.6916 +/- 0.0000 | 0.8592 +/- 0.0000 |  |  |
| `mlp_32x32` | 0.7165 +/- 0.0000 | 0.9106 +/- 0.0000 |  |  |
| `mlp_h64` | 0.9405 +/- 0.0000 | 0.9799 +/- 0.0000 |  |  |
| `mlp_h128` | 0.9401 +/- 0.0000 | 0.9839 +/- 0.0000 |  |  |
| `mlp_h64_64` | 0.9280 +/- 0.0000 | 0.9589 +/- 0.0000 |  |  |
| `upgd_low_noise` | 1.0053 +/- 0.0000 | 0.9631 +/- 0.0000 |  |  |
| `dynamic_sparse` | 0.9894 +/- 0.0000 | 0.9744 +/- 0.0000 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.4622 +/- 0.0000; wins/losses/ties 1/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

# Step 2 Conclusive Learner Candidate

Protocol: 10 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0448 +/- 0.0081 | 0.0782 +/- 0.0062 |  |  |
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

`final_window_mse` conclusive-vs-best-MLP diff: +0.0195 +/- 0.0054; wins/losses/ties 9/1/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

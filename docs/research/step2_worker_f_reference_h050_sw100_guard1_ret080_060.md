# Step 2 Conclusive Learner Candidate

Protocol: 10 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0030 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9903 +/- 0.0008 | 0.4724 +/- 0.0115 |
| `recursive_features` | 0.0174 +/- 0.0007 | 0.0148 +/- 0.0002 | 0.9423 +/- 0.0044 | 0.4724 +/- 0.0115 |
| `polynomial_features` | 0.0378 +/- 0.0028 | 0.0329 +/- 0.0011 | 0.9210 +/- 0.0056 | 0.1440 +/- 0.0127 |
| `fourier_features` | 0.0058 +/- 0.0001 | 0.0062 +/- 0.0000 | 0.9723 +/- 0.0016 | 0.1063 +/- 0.0032 |
| `tanh_random_features` | 0.0057 +/- 0.0002 | 0.0057 +/- 0.0000 | 0.9800 +/- 0.0014 | 0.1499 +/- 0.0108 |
| `mlp_32x32_s01_no_ln` | 0.0047 +/- 0.0002 | 0.0065 +/- 0.0001 | 0.9793 +/- 0.0004 | 0.1006 +/- 0.0004 |
| `mlp_64x64_s01_no_ln` | 0.0060 +/- 0.0004 | 0.0076 +/- 0.0002 | 0.9750 +/- 0.0007 | 0.1006 +/- 0.0004 |
| `mlp_32x32` | 0.0110 +/- 0.0005 | 0.0139 +/- 0.0003 | 0.9433 +/- 0.0033 | 0.1006 +/- 0.0004 |
| `mlp_h64` | 0.0052 +/- 0.0001 | 0.0091 +/- 0.0001 | 0.9860 +/- 0.0007 | 0.1154 +/- 0.0045 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0117 +/- 0.0001 | 0.9857 +/- 0.0012 | 0.1215 +/- 0.0058 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0063 +/- 0.0001 | 0.9903 +/- 0.0008 | 0.1006 +/- 0.0004 |
| `upgd_low_noise` | 0.0139 +/- 0.0004 | 0.0215 +/- 0.0004 | 0.9427 +/- 0.0025 | 0.1516 +/- 0.0134 |
| `dynamic_sparse` | 0.0277 +/- 0.0009 | 0.0387 +/- 0.0006 | 0.9287 +/- 0.0038 | 0.2206 +/- 0.0126 |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0000 +/- 0.0000; wins/losses/ties 0/0/10.
`test_accuracy` conclusive-vs-best-MLP diff: +0.3492 +/- 0.0080; wins/losses/ties 10/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 0/1 configured datasets.

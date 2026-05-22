# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0334 +/- 0.0010 | 0.0470 +/- 0.0007 | 0.8800 +/- 0.0135 | 0.9140 +/- 0.0091 |
| `recursive_features` | 0.0500 +/- 0.0013 | 0.0574 +/- 0.0003 | 0.7789 +/- 0.0194 | 0.8763 +/- 0.0120 |
| `polynomial_features` | 0.2149 +/- 0.0056 | 0.1952 +/- 0.0009 | 0.2700 +/- 0.0077 | 0.2455 +/- 0.0217 |
| `fourier_features` | 0.0891 +/- 0.0002 | 0.0906 +/- 0.0002 | 0.2867 +/- 0.0096 | 0.2690 +/- 0.0392 |
| `tanh_random_features` | 0.0479 +/- 0.0008 | 0.0552 +/- 0.0007 | 0.8222 +/- 0.0146 | 0.8633 +/- 0.0041 |
| `mlp_32x32_s01_no_ln` | 0.0405 +/- 0.0012 | 0.0601 +/- 0.0005 | 0.8067 +/- 0.0167 | 0.8448 +/- 0.0257 |
| `mlp_64x64_s01_no_ln` | 0.0471 +/- 0.0009 | 0.0634 +/- 0.0002 | 0.7733 +/- 0.0117 | 0.8510 +/- 0.0217 |
| `mlp_32x32` | 0.0503 +/- 0.0028 | 0.0663 +/- 0.0013 | 0.7233 +/- 0.0271 | 0.8330 +/- 0.0361 |
| `mlp_h64` | 0.0396 +/- 0.0012 | 0.0560 +/- 0.0007 | 0.8478 +/- 0.0135 | 0.8874 +/- 0.0100 |
| `mlp_h128` | 0.0430 +/- 0.0018 | 0.0575 +/- 0.0017 | 0.8467 +/- 0.0150 | 0.9066 +/- 0.0111 |
| `mlp_h64_64` | 0.0411 +/- 0.0018 | 0.0605 +/- 0.0004 | 0.8311 +/- 0.0183 | 0.8881 +/- 0.0166 |
| `upgd_low_noise` | 0.0559 +/- 0.0016 | 0.0743 +/- 0.0011 | 0.6700 +/- 0.0184 | 0.8429 +/- 0.0100 |
| `dynamic_sparse` | 0.0629 +/- 0.0010 | 0.0877 +/- 0.0016 | 0.6767 +/- 0.0117 | 0.8163 +/- 0.0070 |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0059 +/- 0.0006; wins/losses/ties 3/0/0.
`test_accuracy` conclusive-vs-best-MLP diff: +0.0037 +/- 0.0000; wins/losses/ties 3/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

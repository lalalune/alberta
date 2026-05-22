# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0029 +/- 0.0000 | 0.0062 +/- 0.0003 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |
| `recursive_features` | 0.0193 +/- 0.0005 | 0.0158 +/- 0.0003 | 0.9444 +/- 0.0091 | 0.5077 +/- 0.0252 |
| `polynomial_features` | 0.0502 +/- 0.0054 | 0.0394 +/- 0.0023 | 0.8600 +/- 0.0084 | 0.2041 +/- 0.0049 |
| `mlp_32x32_s01_no_ln` | 0.0054 +/- 0.0006 | 0.0067 +/- 0.0004 | 0.9756 +/- 0.0062 | 0.1002 +/- 0.0000 |
| `mlp_64x64_s01_no_ln` | 0.0072 +/- 0.0007 | 0.0076 +/- 0.0002 | 0.9756 +/- 0.0011 | 0.1002 +/- 0.0000 |
| `mlp_32x32` | 0.0116 +/- 0.0002 | 0.0142 +/- 0.0001 | 0.9467 +/- 0.0033 | 0.1002 +/- 0.0000 |
| `mlp_h64` | 0.0053 +/- 0.0002 | 0.0092 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1379 +/- 0.0152 |
| `mlp_h128` | 0.0076 +/- 0.0003 | 0.0113 +/- 0.0004 | 0.9867 +/- 0.0000 | 0.1169 +/- 0.0065 |
| `mlp_h64_64` | 0.0029 +/- 0.0000 | 0.0063 +/- 0.0003 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |
| `upgd_low_noise` | 0.0137 +/- 0.0005 | 0.0207 +/- 0.0007 | 0.9489 +/- 0.0011 | 0.2004 +/- 0.0171 |
| `dynamic_sparse` | 0.0266 +/- 0.0010 | 0.0403 +/- 0.0006 | 0.9378 +/- 0.0044 | 0.2282 +/- 0.0270 |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0000 +/- 0.0000; wins/losses/ties 0/0/3.
`test_accuracy` conclusive-vs-best-MLP diff: -0.0383 +/- 0.0145; wins/losses/ties 0/3/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 0/1 configured datasets.

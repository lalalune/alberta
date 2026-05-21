# Step 2 Conclusive Learner Candidate

Protocol: 2 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0227 +/- 0.0012 | 0.0383 +/- 0.0002 | 0.9267 +/- 0.0067 | 0.9490 +/- 0.0028 |
| `recursive_features` | 0.0371 +/- 0.0001 | 0.0414 +/- 0.0001 | 0.9150 +/- 0.0017 | 0.9091 +/- 0.0167 |
| `polynomial_features` | 0.2149 +/- 0.0119 | 0.1898 +/- 0.0051 | 0.2800 +/- 0.0067 | 0.2718 +/- 0.0158 |
| `fourier_features` | 0.0874 +/- 0.0005 | 0.0872 +/- 0.0001 | 0.3200 +/- 0.0033 | 0.2653 +/- 0.0705 |
| `tanh_random_features` | 0.0344 +/- 0.0011 | 0.0396 +/- 0.0005 | 0.9117 +/- 0.0117 | 0.9286 +/- 0.0083 |
| `mlp_32x32_s01_no_ln` | 0.0298 +/- 0.0031 | 0.0459 +/- 0.0003 | 0.8933 +/- 0.0167 | 0.8952 +/- 0.0083 |
| `mlp_64x64_s01_no_ln` | 0.0310 +/- 0.0021 | 0.0452 +/- 0.0003 | 0.9067 +/- 0.0333 | 0.9091 +/- 0.0074 |
| `mlp_32x32` | 0.0306 +/- 0.0018 | 0.0482 +/- 0.0000 | 0.8717 +/- 0.0183 | 0.9128 +/- 0.0093 |
| `mlp_h64` | 0.0333 +/- 0.0008 | 0.0446 +/- 0.0002 | 0.9017 +/- 0.0183 | 0.9017 +/- 0.0130 |
| `mlp_h128` | 0.0317 +/- 0.0016 | 0.0433 +/- 0.0009 | 0.9133 +/- 0.0100 | 0.9369 +/- 0.0019 |
| `mlp_h64_64` | 0.0336 +/- 0.0000 | 0.0474 +/- 0.0003 | 0.8950 +/- 0.0150 | 0.9119 +/- 0.0195 |
| `upgd_low_noise` | 0.0328 +/- 0.0008 | 0.0511 +/- 0.0015 | 0.8900 +/- 0.0300 | 0.9202 +/- 0.0130 |
| `dynamic_sparse` | 0.0398 +/- 0.0000 | 0.0666 +/- 0.0016 | 0.8683 +/- 0.0050 | 0.8970 +/- 0.0139 |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0068 +/- 0.0016; wins/losses/ties 2/0/0.
`test_accuracy` conclusive-vs-best-MLP diff: +0.0121 +/- 0.0009; wins/losses/ties 2/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

# Step 2 Conclusive Learner Candidate

Protocol: 1 seeds, 200 steps, final window 50; route loss decay=0.99, warmup=50, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0538 +/- 0.0000 | 0.3034 +/- 0.0000 |  |  |
| `recursive_features` | 0.0495 +/- 0.0000 | 0.2423 +/- 0.0000 |  |  |
| `mlp_32x32_s01_no_ln` | 0.8255 +/- 0.0000 | 1.0212 +/- 0.0000 |  |  |
| `mlp_64x64_s01_no_ln` | 0.7800 +/- 0.0000 | 1.0289 +/- 0.0000 |  |  |
| `mlp_32x32` | 0.8495 +/- 0.0000 | 1.0695 +/- 0.0000 |  |  |
| `mlp_h64` | 0.9087 +/- 0.0000 | 1.1771 +/- 0.0000 |  |  |
| `mlp_h128` | 0.9609 +/- 0.0000 | 1.1667 +/- 0.0000 |  |  |
| `mlp_h64_64` | 0.9290 +/- 0.0000 | 1.1302 +/- 0.0000 |  |  |
| `upgd_low_noise` | 0.9166 +/- 0.0000 | 1.0803 +/- 0.0000 |  |  |
| `dynamic_sparse` | 0.9297 +/- 0.0000 | 1.0918 +/- 0.0000 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.7262 +/- 0.0000; wins/losses/ties 1/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

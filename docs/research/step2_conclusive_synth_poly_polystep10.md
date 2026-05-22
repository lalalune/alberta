# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.7955 +/- 0.5234 | 0.8874 +/- 0.2020 |  |  |
| `recursive_features` | 0.9570 +/- 0.6651 | 0.9717 +/- 0.2227 |  |  |
| `polynomial_features` | 1.0836 +/- 0.3348 | 0.9989 +/- 0.2282 |  |  |
| `mlp_32x32_s01_no_ln` | 0.9170 +/- 0.6410 | 0.9710 +/- 0.2296 |  |  |
| `mlp_64x64_s01_no_ln` | 0.9069 +/- 0.6342 | 0.9606 +/- 0.2289 |  |  |
| `mlp_32x32` | 0.9112 +/- 0.6415 | 0.9690 +/- 0.2285 |  |  |
| `mlp_h64` | 0.9652 +/- 0.6492 | 1.0431 +/- 0.2389 |  |  |
| `mlp_h128` | 0.9715 +/- 0.6520 | 1.0469 +/- 0.2386 |  |  |
| `mlp_h64_64` | 0.9488 +/- 0.6470 | 1.0287 +/- 0.2384 |  |  |
| `upgd_low_noise` | 0.9141 +/- 0.6323 | 0.9790 +/- 0.2251 |  |  |
| `dynamic_sparse` | 0.9183 +/- 0.6356 | 0.9828 +/- 0.2301 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.1073 +/- 0.1130; wins/losses/ties 1/2/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/1 configured datasets.

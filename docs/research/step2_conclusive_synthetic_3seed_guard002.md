# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.02.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.2748 +/- 0.0966 | 0.2010 +/- 0.0306 |  |  |
| `recursive_features` | 0.9415 +/- 0.3487 | 0.6405 +/- 0.1299 |  |  |
| `polynomial_features` | 1.4973 +/- 0.6269 | 0.9227 +/- 0.1611 |  |  |
| `mlp_32x32_s01_no_ln` | 0.5021 +/- 0.2324 | 0.3042 +/- 0.0704 |  |  |
| `mlp_64x64_s01_no_ln` | 0.5400 +/- 0.2153 | 0.3230 +/- 0.0622 |  |  |
| `mlp_32x32` | 0.4905 +/- 0.2032 | 0.3326 +/- 0.0666 |  |  |
| `mlp_h64` | 0.2778 +/- 0.0972 | 0.1823 +/- 0.0299 |  |  |
| `mlp_h128` | 0.2790 +/- 0.0934 | 0.1809 +/- 0.0301 |  |  |
| `mlp_h64_64` | 0.3288 +/- 0.1119 | 0.2196 +/- 0.0320 |  |  |
| `upgd_low_noise` | 0.4105 +/- 0.1421 | 0.2613 +/- 0.0490 |  |  |
| `dynamic_sparse` | 0.3987 +/- 0.1585 | 0.2543 +/- 0.0505 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0001 +/- 0.0045; wins/losses/ties 1/2/0.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 1.0222 +/- 0.2426 | 1.1917 +/- 0.2229 |  |  |
| `recursive_features` | 2.6877 +/- 0.3196 | 2.3972 +/- 0.2927 |  |  |
| `polynomial_features` | 2.1267 +/- 0.2323 | 2.1940 +/- 0.4568 |  |  |
| `mlp_32x32_s01_no_ln` | 1.0744 +/- 0.2295 | 1.2503 +/- 0.2290 |  |  |
| `mlp_64x64_s01_no_ln` | 1.1391 +/- 0.1742 | 1.2931 +/- 0.2577 |  |  |
| `mlp_32x32` | 1.0145 +/- 0.2289 | 1.2036 +/- 0.2291 |  |  |
| `mlp_h64` | 1.1299 +/- 0.2841 | 1.3246 +/- 0.2359 |  |  |
| `mlp_h128` | 1.1646 +/- 0.2642 | 1.3661 +/- 0.2498 |  |  |
| `mlp_h64_64` | 1.1387 +/- 0.2691 | 1.3447 +/- 0.2470 |  |  |
| `upgd_low_noise` | 1.2504 +/- 0.1798 | 1.3509 +/- 0.2760 |  |  |
| `dynamic_sparse` | 1.2517 +/- 0.1790 | 1.3455 +/- 0.2737 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0138 +/- 0.0120; wins/losses/ties 1/2/0.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.8224 +/- 0.5402 | 0.9075 +/- 0.2152 |  |  |
| `recursive_features` | 0.9570 +/- 0.6651 | 0.9717 +/- 0.2227 |  |  |
| `polynomial_features` | 0.8185 +/- 0.5094 | 0.8794 +/- 0.2106 |  |  |
| `mlp_32x32_s01_no_ln` | 0.9170 +/- 0.6410 | 0.9710 +/- 0.2296 |  |  |
| `mlp_64x64_s01_no_ln` | 0.9069 +/- 0.6342 | 0.9606 +/- 0.2289 |  |  |
| `mlp_32x32` | 0.9112 +/- 0.6415 | 0.9690 +/- 0.2285 |  |  |
| `mlp_h64` | 0.9652 +/- 0.6492 | 1.0431 +/- 0.2389 |  |  |
| `mlp_h128` | 0.9715 +/- 0.6520 | 1.0469 +/- 0.2386 |  |  |
| `mlp_h64_64` | 0.9488 +/- 0.6470 | 1.0287 +/- 0.2384 |  |  |
| `upgd_low_noise` | 0.9141 +/- 0.6323 | 0.9790 +/- 0.2251 |  |  |
| `dynamic_sparse` | 0.9183 +/- 0.6356 | 0.9828 +/- 0.2301 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0805 +/- 0.0961; wins/losses/ties 1/2/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/3 configured datasets.

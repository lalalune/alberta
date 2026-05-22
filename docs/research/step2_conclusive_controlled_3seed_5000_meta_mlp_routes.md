# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 5000 steps, final window 1000; route loss decay=0.99, warmup=500, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0103 +/- 0.0012 | 0.0803 +/- 0.0024 |  |  |
| `recursive_features` | 0.0750 +/- 0.0295 | 0.0932 +/- 0.0185 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0193 +/- 0.0002 | 0.1473 +/- 0.0115 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0367 +/- 0.0095 | 0.1623 +/- 0.0200 |  |  |
| `mlp_32x32` | 0.0102 +/- 0.0012 | 0.2058 +/- 0.0105 |  |  |
| `mlp_h64` | 0.0333 +/- 0.0048 | 0.1935 +/- 0.0182 |  |  |
| `mlp_h128` | 0.0469 +/- 0.0041 | 0.2265 +/- 0.0061 |  |  |
| `mlp_h64_64` | 0.0320 +/- 0.0048 | 0.1417 +/- 0.0242 |  |  |
| `upgd_low_noise` | 0.0621 +/- 0.0027 | 0.2840 +/- 0.0171 |  |  |
| `dynamic_sparse` | 0.0516 +/- 0.0054 | 0.2361 +/- 0.0208 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0001 +/- 0.0001; wins/losses/ties 0/1/2.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0438 +/- 0.0069 | 0.1166 +/- 0.0074 |  |  |
| `recursive_features` | 0.0901 +/- 0.0133 | 0.1285 +/- 0.0053 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0768 +/- 0.0136 | 0.1768 +/- 0.0164 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0772 +/- 0.0234 | 0.1694 +/- 0.0229 |  |  |
| `mlp_32x32` | 0.0456 +/- 0.0101 | 0.1737 +/- 0.0184 |  |  |
| `mlp_h64` | 0.1107 +/- 0.0062 | 0.3174 +/- 0.0161 |  |  |
| `mlp_h128` | 0.1566 +/- 0.0116 | 0.3944 +/- 0.0148 |  |  |
| `mlp_h64_64` | 0.3466 +/- 0.0067 | 0.5499 +/- 0.0064 |  |  |
| `upgd_low_noise` | 0.1975 +/- 0.0128 | 0.4390 +/- 0.0182 |  |  |
| `dynamic_sparse` | 0.2036 +/- 0.0178 | 0.4394 +/- 0.0241 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0018 +/- 0.0006; wins/losses/ties 0/3/0.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0181 +/- 0.0011 | 0.0432 +/- 0.0016 |  |  |
| `recursive_features` | 0.2392 +/- 0.0796 | 0.2401 +/- 0.0479 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0210 +/- 0.0015 | 0.0537 +/- 0.0019 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0299 +/- 0.0002 | 0.0591 +/- 0.0010 |  |  |
| `mlp_32x32` | 0.0178 +/- 0.0012 | 0.0499 +/- 0.0010 |  |  |
| `mlp_h64` | 0.0214 +/- 0.0002 | 0.0532 +/- 0.0015 |  |  |
| `mlp_h128` | 0.0277 +/- 0.0008 | 0.0638 +/- 0.0018 |  |  |
| `mlp_h64_64` | 0.0393 +/- 0.0025 | 0.0805 +/- 0.0008 |  |  |
| `upgd_low_noise` | 0.0285 +/- 0.0016 | 0.0688 +/- 0.0024 |  |  |
| `dynamic_sparse` | 0.0374 +/- 0.0024 | 0.0786 +/- 0.0038 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0004 +/- 0.0003; wins/losses/ties 1/2/0.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.1705 +/- 0.0342 | 0.3421 +/- 0.0345 |  |  |
| `recursive_features` | 0.5327 +/- 0.0513 | 0.4248 +/- 0.0318 |  |  |
| `mlp_32x32_s01_no_ln` | 0.1852 +/- 0.0212 | 0.4521 +/- 0.0291 |  |  |
| `mlp_64x64_s01_no_ln` | 0.2329 +/- 0.0237 | 0.4642 +/- 0.0199 |  |  |
| `mlp_32x32` | 0.1500 +/- 0.0251 | 0.4252 +/- 0.0424 |  |  |
| `mlp_h64` | 0.3385 +/- 0.0682 | 0.6773 +/- 0.0215 |  |  |
| `mlp_h128` | 0.3786 +/- 0.0661 | 0.7441 +/- 0.0236 |  |  |
| `mlp_h64_64` | 0.5554 +/- 0.0900 | 0.8777 +/- 0.0346 |  |  |
| `upgd_low_noise` | 0.5291 +/- 0.0904 | 0.8677 +/- 0.0323 |  |  |
| `dynamic_sparse` | 0.7035 +/- 0.0972 | 0.9718 +/- 0.0272 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0206 +/- 0.0168; wins/losses/ties 0/3/0.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0376 +/- 0.0026 | 0.0667 +/- 0.0053 |  |  |
| `recursive_features` | 0.1002 +/- 0.0050 | 0.1152 +/- 0.0184 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0391 +/- 0.0032 | 0.0710 +/- 0.0072 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0376 +/- 0.0026 | 0.0678 +/- 0.0056 |  |  |
| `mlp_32x32` | 0.0406 +/- 0.0032 | 0.0725 +/- 0.0070 |  |  |
| `mlp_h64` | 0.0517 +/- 0.0041 | 0.0907 +/- 0.0065 |  |  |
| `mlp_h128` | 0.0534 +/- 0.0042 | 0.0939 +/- 0.0070 |  |  |
| `mlp_h64_64` | 0.0531 +/- 0.0038 | 0.0989 +/- 0.0065 |  |  |
| `upgd_low_noise` | 0.0547 +/- 0.0046 | 0.0950 +/- 0.0077 |  |  |
| `dynamic_sparse` | 0.0574 +/- 0.0030 | 0.0967 +/- 0.0075 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0001 +/- 0.0007; wins/losses/ties 1/2/0.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0906 +/- 0.0183 | 0.1834 +/- 0.0103 |  |  |
| `recursive_features` | 0.0868 +/- 0.0225 | 0.1069 +/- 0.0196 |  |  |
| `mlp_32x32_s01_no_ln` | 0.2442 +/- 0.0575 | 0.5147 +/- 0.0580 |  |  |
| `mlp_64x64_s01_no_ln` | 0.3034 +/- 0.0424 | 0.5529 +/- 0.0412 |  |  |
| `mlp_32x32` | 0.2357 +/- 0.0354 | 0.5743 +/- 0.0454 |  |  |
| `mlp_h64` | 0.7337 +/- 0.0995 | 1.0280 +/- 0.0524 |  |  |
| `mlp_h128` | 0.7783 +/- 0.1097 | 1.0624 +/- 0.0687 |  |  |
| `mlp_h64_64` | 0.5767 +/- 0.0326 | 0.7881 +/- 0.0295 |  |  |
| `upgd_low_noise` | 0.8696 +/- 0.0991 | 1.0720 +/- 0.0578 |  |  |
| `dynamic_sparse` | 0.9824 +/- 0.0751 | 1.1171 +/- 0.0492 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.1361 +/- 0.0244; wins/losses/ties 3/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 1/6 configured datasets.

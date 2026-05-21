# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 2000 steps, final window 500; route loss decay=0.95, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0519 +/- 0.0051 | 0.1594 +/- 0.0160 |  |  |
| `recursive_features` | 0.0608 +/- 0.0158 | 0.1418 +/- 0.0259 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0551 +/- 0.0078 | 0.3251 +/- 0.0201 |  |  |
| `mlp_64x64_s01_no_ln` | 0.1181 +/- 0.0351 | 0.3457 +/- 0.0386 |  |  |
| `mlp_32x32` | 0.3342 +/- 0.0539 | 0.4472 +/- 0.0212 |  |  |
| `mlp_h64` | 0.1350 +/- 0.0243 | 0.3939 +/- 0.0240 |  |  |
| `mlp_h128` | 0.2093 +/- 0.0141 | 0.4607 +/- 0.0117 |  |  |
| `mlp_h64_64` | 0.1048 +/- 0.0267 | 0.2851 +/- 0.0506 |  |  |
| `upgd_low_noise` | 0.3080 +/- 0.0164 | 0.5305 +/- 0.0217 |  |  |
| `dynamic_sparse` | 0.2366 +/- 0.0541 | 0.4837 +/- 0.0319 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0032 +/- 0.0044; wins/losses/ties 2/1/0.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.1068 +/- 0.0172 | 0.2478 +/- 0.0161 |  |  |
| `recursive_features` | 0.1126 +/- 0.0211 | 0.1883 +/- 0.0015 |  |  |
| `mlp_32x32_s01_no_ln` | 0.1828 +/- 0.0668 | 0.2853 +/- 0.0386 |  |  |
| `mlp_64x64_s01_no_ln` | 0.1536 +/- 0.0344 | 0.2889 +/- 0.0381 |  |  |
| `mlp_32x32` | 0.1552 +/- 0.0319 | 0.3325 +/- 0.0348 |  |  |
| `mlp_h64` | 0.3667 +/- 0.0386 | 0.5420 +/- 0.0103 |  |  |
| `mlp_h128` | 0.5128 +/- 0.0859 | 0.6233 +/- 0.0360 |  |  |
| `mlp_h64_64` | 0.6599 +/- 0.0790 | 0.7141 +/- 0.0106 |  |  |
| `upgd_low_noise` | 0.5333 +/- 0.0783 | 0.6550 +/- 0.0526 |  |  |
| `dynamic_sparse` | 0.5326 +/- 0.0893 | 0.6651 +/- 0.0295 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0333 +/- 0.0222; wins/losses/ties 2/1/0.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0494 +/- 0.0065 | 0.0847 +/- 0.0020 |  |  |
| `recursive_features` | 0.1516 +/- 0.0194 | 0.2057 +/- 0.0066 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0569 +/- 0.0077 | 0.0932 +/- 0.0067 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0623 +/- 0.0028 | 0.0891 +/- 0.0022 |  |  |
| `mlp_32x32` | 0.0440 +/- 0.0029 | 0.0892 +/- 0.0019 |  |  |
| `mlp_h64` | 0.0584 +/- 0.0042 | 0.0922 +/- 0.0022 |  |  |
| `mlp_h128` | 0.0749 +/- 0.0039 | 0.1053 +/- 0.0023 |  |  |
| `mlp_h64_64` | 0.1004 +/- 0.0018 | 0.1206 +/- 0.0026 |  |  |
| `upgd_low_noise` | 0.0803 +/- 0.0069 | 0.1118 +/- 0.0033 |  |  |
| `dynamic_sparse` | 0.0908 +/- 0.0080 | 0.1219 +/- 0.0030 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0054 +/- 0.0037; wins/losses/ties 1/2/0.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.6023 +/- 0.2395 | 0.5600 +/- 0.0967 |  |  |
| `recursive_features` | 0.9916 +/- 0.3458 | 0.6653 +/- 0.1980 |  |  |
| `mlp_32x32_s01_no_ln` | 0.6426 +/- 0.2536 | 0.7345 +/- 0.0909 |  |  |
| `mlp_64x64_s01_no_ln` | 0.5974 +/- 0.1976 | 0.7058 +/- 0.0818 |  |  |
| `mlp_32x32` | 0.6262 +/- 0.2611 | 0.7239 +/- 0.1032 |  |  |
| `mlp_h64` | 0.9262 +/- 0.2543 | 1.0214 +/- 0.0962 |  |  |
| `mlp_h128` | 1.0116 +/- 0.2441 | 1.0846 +/- 0.0812 |  |  |
| `mlp_h64_64` | 1.2093 +/- 0.2786 | 1.1561 +/- 0.1012 |  |  |
| `upgd_low_noise` | 1.1753 +/- 0.2903 | 1.1644 +/- 0.1145 |  |  |
| `dynamic_sparse` | 1.2017 +/- 0.2743 | 1.1801 +/- 0.1122 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0434 +/- 0.0292; wins/losses/ties 1/2/0.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0340 +/- 0.0091 | 0.0766 +/- 0.0105 |  |  |
| `recursive_features` | 0.0854 +/- 0.0067 | 0.1142 +/- 0.0065 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0338 +/- 0.0086 | 0.0713 +/- 0.0084 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0330 +/- 0.0086 | 0.0679 +/- 0.0091 |  |  |
| `mlp_32x32` | 0.0326 +/- 0.0082 | 0.0737 +/- 0.0107 |  |  |
| `mlp_h64` | 0.0524 +/- 0.0082 | 0.0984 +/- 0.0087 |  |  |
| `mlp_h128` | 0.0567 +/- 0.0089 | 0.1021 +/- 0.0091 |  |  |
| `mlp_h64_64` | 0.0688 +/- 0.0089 | 0.1172 +/- 0.0082 |  |  |
| `upgd_low_noise` | 0.0567 +/- 0.0100 | 0.1052 +/- 0.0088 |  |  |
| `dynamic_sparse` | 0.0590 +/- 0.0091 | 0.1063 +/- 0.0091 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0017 +/- 0.0009; wins/losses/ties 0/3/0.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.1094 +/- 0.0276 | 0.1990 +/- 0.0157 |  |  |
| `recursive_features` | 0.1044 +/- 0.0302 | 0.1102 +/- 0.0157 |  |  |
| `mlp_32x32_s01_no_ln` | 0.7566 +/- 0.2332 | 0.7825 +/- 0.0968 |  |  |
| `mlp_64x64_s01_no_ln` | 0.6368 +/- 0.2407 | 0.7545 +/- 0.1235 |  |  |
| `mlp_32x32` | 0.8009 +/- 0.2421 | 0.8850 +/- 0.1111 |  |  |
| `mlp_h64` | 1.2098 +/- 0.1905 | 1.1374 +/- 0.0945 |  |  |
| `mlp_h128` | 1.2303 +/- 0.2045 | 1.1490 +/- 0.0966 |  |  |
| `mlp_h64_64` | 0.9263 +/- 0.1929 | 0.9159 +/- 0.0807 |  |  |
| `upgd_low_noise` | 1.2090 +/- 0.2029 | 1.1121 +/- 0.0959 |  |  |
| `dynamic_sparse` | 1.2196 +/- 0.2048 | 1.1231 +/- 0.1000 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.5274 +/- 0.2139; wins/losses/ties 3/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 3/6 configured datasets.

# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 2000 steps, final window 500; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0272 +/- 0.0011 | 0.1478 +/- 0.0174 |  |  |
| `recursive_features` | 0.0608 +/- 0.0158 | 0.1418 +/- 0.0259 |  |  |
| `polynomial_features` | 0.0270 +/- 0.0013 | 0.1676 +/- 0.0046 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0525 +/- 0.0129 | 0.3154 +/- 0.0359 |  |  |
| `mlp_64x64_s01_no_ln` | 0.1212 +/- 0.0261 | 0.3678 +/- 0.0248 |  |  |
| `mlp_32x32` | 0.3702 +/- 0.0067 | 0.4631 +/- 0.0059 |  |  |
| `mlp_h64` | 0.1699 +/- 0.0109 | 0.4302 +/- 0.0023 |  |  |
| `mlp_h128` | 0.2171 +/- 0.0406 | 0.4811 +/- 0.0277 |  |  |
| `mlp_h64_64` | 0.0901 +/- 0.0065 | 0.2798 +/- 0.0117 |  |  |
| `upgd_low_noise` | 0.2445 +/- 0.0195 | 0.4996 +/- 0.0184 |  |  |
| `dynamic_sparse` | 0.2840 +/- 0.0682 | 0.5272 +/- 0.0292 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0253 +/- 0.0131; wins/losses/ties 3/0/0.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0060 +/- 0.0015 | 0.1673 +/- 0.0189 |  |  |
| `recursive_features` | 0.1126 +/- 0.0211 | 0.1883 +/- 0.0015 |  |  |
| `polynomial_features` | 0.0060 +/- 0.0015 | 0.1535 +/- 0.0152 |  |  |
| `mlp_32x32_s01_no_ln` | 0.1502 +/- 0.0316 | 0.2797 +/- 0.0241 |  |  |
| `mlp_64x64_s01_no_ln` | 0.1548 +/- 0.0455 | 0.2952 +/- 0.0255 |  |  |
| `mlp_32x32` | 0.1392 +/- 0.0471 | 0.3029 +/- 0.0216 |  |  |
| `mlp_h64` | 0.3557 +/- 0.0599 | 0.5362 +/- 0.0262 |  |  |
| `mlp_h128` | 0.5235 +/- 0.0856 | 0.6153 +/- 0.0326 |  |  |
| `mlp_h64_64` | 0.6267 +/- 0.0847 | 0.7157 +/- 0.0411 |  |  |
| `upgd_low_noise` | 0.5055 +/- 0.0554 | 0.6298 +/- 0.0224 |  |  |
| `dynamic_sparse` | 0.5253 +/- 0.0637 | 0.6453 +/- 0.0239 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.1266 +/- 0.0389; wins/losses/ties 3/0/0.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0392 +/- 0.0056 | 0.0771 +/- 0.0038 |  |  |
| `recursive_features` | 0.1516 +/- 0.0194 | 0.2057 +/- 0.0066 |  |  |
| `polynomial_features` | 0.0767 +/- 0.0200 | 0.1694 +/- 0.0062 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0507 +/- 0.0116 | 0.0795 +/- 0.0033 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0580 +/- 0.0042 | 0.0888 +/- 0.0020 |  |  |
| `mlp_32x32` | 0.0500 +/- 0.0066 | 0.0992 +/- 0.0065 |  |  |
| `mlp_h64` | 0.0517 +/- 0.0030 | 0.0879 +/- 0.0012 |  |  |
| `mlp_h128` | 0.0807 +/- 0.0047 | 0.1073 +/- 0.0020 |  |  |
| `mlp_h64_64` | 0.0911 +/- 0.0059 | 0.1160 +/- 0.0040 |  |  |
| `upgd_low_noise` | 0.0790 +/- 0.0054 | 0.1116 +/- 0.0044 |  |  |
| `dynamic_sparse` | 0.0808 +/- 0.0064 | 0.1112 +/- 0.0034 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0049 +/- 0.0009; wins/losses/ties 3/0/0.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0440 +/- 0.0292 | 0.2137 +/- 0.0215 |  |  |
| `recursive_features` | 0.9916 +/- 0.3458 | 0.6653 +/- 0.1980 |  |  |
| `polynomial_features` | 0.0440 +/- 0.0292 | 0.1416 +/- 0.0180 |  |  |
| `mlp_32x32_s01_no_ln` | 0.6463 +/- 0.2411 | 0.7511 +/- 0.0887 |  |  |
| `mlp_64x64_s01_no_ln` | 0.6681 +/- 0.2559 | 0.7113 +/- 0.0899 |  |  |
| `mlp_32x32` | 0.6313 +/- 0.2803 | 0.7448 +/- 0.1064 |  |  |
| `mlp_h64` | 0.8759 +/- 0.2432 | 0.9936 +/- 0.0852 |  |  |
| `mlp_h128` | 1.0543 +/- 0.2451 | 1.1149 +/- 0.0889 |  |  |
| `mlp_h64_64` | 1.1960 +/- 0.3119 | 1.1670 +/- 0.1133 |  |  |
| `upgd_low_noise` | 1.1740 +/- 0.2634 | 1.1791 +/- 0.1011 |  |  |
| `dynamic_sparse` | 1.2108 +/- 0.2782 | 1.1874 +/- 0.1132 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.5659 +/- 0.2305; wins/losses/ties 3/0/0.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0306 +/- 0.0098 | 0.0706 +/- 0.0098 |  |  |
| `recursive_features` | 0.0854 +/- 0.0067 | 0.1142 +/- 0.0065 |  |  |
| `polynomial_features` | 0.0251 +/- 0.0053 | 0.0872 +/- 0.0090 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0344 +/- 0.0088 | 0.0694 +/- 0.0091 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0330 +/- 0.0089 | 0.0674 +/- 0.0092 |  |  |
| `mlp_32x32` | 0.0334 +/- 0.0089 | 0.0755 +/- 0.0110 |  |  |
| `mlp_h64` | 0.0526 +/- 0.0092 | 0.0984 +/- 0.0105 |  |  |
| `mlp_h128` | 0.0561 +/- 0.0087 | 0.1021 +/- 0.0099 |  |  |
| `mlp_h64_64` | 0.0672 +/- 0.0122 | 0.1160 +/- 0.0077 |  |  |
| `upgd_low_noise` | 0.0583 +/- 0.0086 | 0.1066 +/- 0.0090 |  |  |
| `dynamic_sparse` | 0.0562 +/- 0.0082 | 0.1031 +/- 0.0097 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0023 +/- 0.0018; wins/losses/ties 3/0/0.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0392 +/- 0.0139 | 0.1572 +/- 0.0145 |  |  |
| `recursive_features` | 0.1044 +/- 0.0302 | 0.1102 +/- 0.0157 |  |  |
| `polynomial_features` | 0.0336 +/- 0.0119 | 0.2196 +/- 0.0149 |  |  |
| `mlp_32x32_s01_no_ln` | 0.8245 +/- 0.1764 | 0.8343 +/- 0.0702 |  |  |
| `mlp_64x64_s01_no_ln` | 0.6956 +/- 0.2223 | 0.7906 +/- 0.1084 |  |  |
| `mlp_32x32` | 0.8142 +/- 0.2225 | 0.8639 +/- 0.0946 |  |  |
| `mlp_h64` | 1.1935 +/- 0.2148 | 1.1303 +/- 0.1011 |  |  |
| `mlp_h128` | 1.2280 +/- 0.2073 | 1.1483 +/- 0.0980 |  |  |
| `mlp_h64_64` | 0.9303 +/- 0.2128 | 0.9450 +/- 0.1119 |  |  |
| `upgd_low_noise` | 1.2081 +/- 0.2034 | 1.1149 +/- 0.0975 |  |  |
| `dynamic_sparse` | 1.2109 +/- 0.1970 | 1.1196 +/- 0.0947 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.6516 +/- 0.2084; wins/losses/ties 3/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 6/6 configured datasets.

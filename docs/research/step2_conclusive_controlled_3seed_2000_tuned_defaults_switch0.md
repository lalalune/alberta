# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 2000 steps, final window 500; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0051 +/- 0.0002 | 0.1007 +/- 0.0011 |  |  |
| `recursive_features` | 0.0608 +/- 0.0158 | 0.1418 +/- 0.0259 |  |  |
| `polynomial_features` | 0.0051 +/- 0.0002 | 0.1015 +/- 0.0064 |  |  |
| `fourier_features` | 0.0135 +/- 0.0004 | 0.0880 +/- 0.0005 |  |  |
| `tanh_random_features` | 0.5211 +/- 0.0191 | 0.5688 +/- 0.0051 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0521 +/- 0.0031 | 0.3326 +/- 0.0050 |  |  |
| `mlp_64x64_s01_no_ln` | 0.1072 +/- 0.0205 | 0.3476 +/- 0.0154 |  |  |
| `mlp_32x32` | 0.2961 +/- 0.0554 | 0.4345 +/- 0.0179 |  |  |
| `mlp_h64` | 0.1634 +/- 0.0240 | 0.4088 +/- 0.0156 |  |  |
| `mlp_h128` | 0.1364 +/- 0.0287 | 0.4021 +/- 0.0250 |  |  |
| `mlp_h64_64` | 0.1063 +/- 0.0057 | 0.2926 +/- 0.0148 |  |  |
| `upgd_low_noise` | 0.3342 +/- 0.0622 | 0.5612 +/- 0.0351 |  |  |
| `dynamic_sparse` | 0.3468 +/- 0.0317 | 0.5767 +/- 0.0159 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0470 +/- 0.0033; wins/losses/ties 3/0/0.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0077 +/- 0.0015 | 0.1575 +/- 0.0141 |  |  |
| `recursive_features` | 0.1126 +/- 0.0211 | 0.1883 +/- 0.0015 |  |  |
| `polynomial_features` | 0.0077 +/- 0.0015 | 0.0874 +/- 0.0081 |  |  |
| `fourier_features` | 1.4979 +/- 0.1396 | 1.4648 +/- 0.0413 |  |  |
| `tanh_random_features` | 0.3851 +/- 0.0652 | 0.6213 +/- 0.0312 |  |  |
| `mlp_32x32_s01_no_ln` | 0.1259 +/- 0.0154 | 0.2703 +/- 0.0206 |  |  |
| `mlp_64x64_s01_no_ln` | 0.1399 +/- 0.0171 | 0.2847 +/- 0.0124 |  |  |
| `mlp_32x32` | 0.1203 +/- 0.0287 | 0.3192 +/- 0.0096 |  |  |
| `mlp_h64` | 0.3833 +/- 0.0566 | 0.5594 +/- 0.0331 |  |  |
| `mlp_h128` | 0.5059 +/- 0.0579 | 0.6124 +/- 0.0229 |  |  |
| `mlp_h64_64` | 0.6448 +/- 0.0903 | 0.7053 +/- 0.0364 |  |  |
| `upgd_low_noise` | 0.5507 +/- 0.0901 | 0.6495 +/- 0.0508 |  |  |
| `dynamic_sparse` | 0.5232 +/- 0.0906 | 0.6928 +/- 0.0270 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0952 +/- 0.0210; wins/losses/ties 3/0/0.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0363 +/- 0.0016 | 0.0752 +/- 0.0028 |  |  |
| `recursive_features` | 0.1516 +/- 0.0194 | 0.2057 +/- 0.0066 |  |  |
| `polynomial_features` | 0.1148 +/- 0.0336 | 0.1433 +/- 0.0076 |  |  |
| `fourier_features` | 0.0635 +/- 0.0012 | 0.0818 +/- 0.0020 |  |  |
| `tanh_random_features` | 0.0690 +/- 0.0069 | 0.1014 +/- 0.0034 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0492 +/- 0.0059 | 0.0931 +/- 0.0074 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0558 +/- 0.0021 | 0.0821 +/- 0.0036 |  |  |
| `mlp_32x32` | 0.0456 +/- 0.0053 | 0.0909 +/- 0.0049 |  |  |
| `mlp_h64` | 0.0584 +/- 0.0037 | 0.0916 +/- 0.0017 |  |  |
| `mlp_h128` | 0.0769 +/- 0.0038 | 0.1052 +/- 0.0011 |  |  |
| `mlp_h64_64` | 0.0904 +/- 0.0037 | 0.1194 +/- 0.0022 |  |  |
| `upgd_low_noise` | 0.0745 +/- 0.0037 | 0.1066 +/- 0.0012 |  |  |
| `dynamic_sparse` | 0.0876 +/- 0.0104 | 0.1224 +/- 0.0062 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0093 +/- 0.0037; wins/losses/ties 3/0/0.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0293 +/- 0.0179 | 0.1702 +/- 0.0167 |  |  |
| `recursive_features` | 0.9916 +/- 0.3458 | 0.6653 +/- 0.1980 |  |  |
| `polynomial_features` | 0.0293 +/- 0.0179 | 0.0794 +/- 0.0134 |  |  |
| `fourier_features` | 0.6370 +/- 0.1684 | 0.6957 +/- 0.0644 |  |  |
| `tanh_random_features` | 0.8263 +/- 0.2397 | 0.9378 +/- 0.0851 |  |  |
| `mlp_32x32_s01_no_ln` | 0.6684 +/- 0.3219 | 0.7243 +/- 0.1140 |  |  |
| `mlp_64x64_s01_no_ln` | 0.6426 +/- 0.2778 | 0.7148 +/- 0.1117 |  |  |
| `mlp_32x32` | 0.6067 +/- 0.2567 | 0.7287 +/- 0.1034 |  |  |
| `mlp_h64` | 0.8567 +/- 0.2335 | 0.9618 +/- 0.0887 |  |  |
| `mlp_h128` | 1.0083 +/- 0.2619 | 1.0730 +/- 0.0929 |  |  |
| `mlp_h64_64` | 1.1535 +/- 0.3015 | 1.1622 +/- 0.1169 |  |  |
| `upgd_low_noise` | 1.1777 +/- 0.3059 | 1.1715 +/- 0.1181 |  |  |
| `dynamic_sparse` | 1.2198 +/- 0.2683 | 1.1944 +/- 0.1036 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.5687 +/- 0.2448; wins/losses/ties 3/0/0.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0219 +/- 0.0064 | 0.0649 +/- 0.0119 |  |  |
| `recursive_features` | 0.0854 +/- 0.0067 | 0.1142 +/- 0.0065 |  |  |
| `polynomial_features` | 0.0208 +/- 0.0055 | 0.0620 +/- 0.0061 |  |  |
| `fourier_features` | 0.0345 +/- 0.0066 | 0.0726 +/- 0.0118 |  |  |
| `tanh_random_features` | 0.0369 +/- 0.0076 | 0.0748 +/- 0.0118 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0349 +/- 0.0090 | 0.0700 +/- 0.0108 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0335 +/- 0.0084 | 0.0686 +/- 0.0101 |  |  |
| `mlp_32x32` | 0.0329 +/- 0.0085 | 0.0758 +/- 0.0118 |  |  |
| `mlp_h64` | 0.0523 +/- 0.0103 | 0.0989 +/- 0.0095 |  |  |
| `mlp_h128` | 0.0560 +/- 0.0098 | 0.1010 +/- 0.0077 |  |  |
| `mlp_h64_64` | 0.0697 +/- 0.0091 | 0.1192 +/- 0.0084 |  |  |
| `upgd_low_noise` | 0.0572 +/- 0.0089 | 0.1053 +/- 0.0111 |  |  |
| `dynamic_sparse` | 0.0572 +/- 0.0099 | 0.1045 +/- 0.0100 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0110 +/- 0.0033; wins/losses/ties 3/0/0.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0097 +/- 0.0035 | 0.1266 +/- 0.0140 |  |  |
| `recursive_features` | 0.1044 +/- 0.0302 | 0.1102 +/- 0.0157 |  |  |
| `polynomial_features` | 0.0097 +/- 0.0035 | 0.1013 +/- 0.0097 |  |  |
| `fourier_features` | 1.2721 +/- 0.1767 | 1.1696 +/- 0.0928 |  |  |
| `tanh_random_features` | 0.7338 +/- 0.2283 | 0.8525 +/- 0.1263 |  |  |
| `mlp_32x32_s01_no_ln` | 0.8111 +/- 0.2490 | 0.8212 +/- 0.1214 |  |  |
| `mlp_64x64_s01_no_ln` | 0.7010 +/- 0.1596 | 0.7784 +/- 0.0756 |  |  |
| `mlp_32x32` | 0.7779 +/- 0.2253 | 0.8815 +/- 0.1037 |  |  |
| `mlp_h64` | 1.1885 +/- 0.2157 | 1.1305 +/- 0.0991 |  |  |
| `mlp_h128` | 1.2310 +/- 0.2012 | 1.1498 +/- 0.0965 |  |  |
| `mlp_h64_64` | 0.9130 +/- 0.2159 | 0.9142 +/- 0.1053 |  |  |
| `upgd_low_noise` | 1.1970 +/- 0.2110 | 1.1069 +/- 0.0988 |  |  |
| `dynamic_sparse` | 1.2180 +/- 0.1998 | 1.1195 +/- 0.0953 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.6389 +/- 0.1810; wins/losses/ties 3/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 6/6 configured datasets.

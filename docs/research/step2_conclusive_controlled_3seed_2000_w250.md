# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 2000 steps, final window 500; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0635 +/- 0.0180 | 0.1678 +/- 0.0267 |  |  |
| `recursive_features` | 0.0608 +/- 0.0158 | 0.1418 +/- 0.0259 |  |  |
| `mlp_32x32` | 0.3229 +/- 0.0654 | 0.4454 +/- 0.0282 |  |  |
| `mlp_h64` | 0.1691 +/- 0.0075 | 0.4170 +/- 0.0115 |  |  |
| `mlp_h128` | 0.1931 +/- 0.0337 | 0.4446 +/- 0.0291 |  |  |
| `mlp_h64_64` | 0.0927 +/- 0.0053 | 0.2682 +/- 0.0221 |  |  |
| `upgd_low_noise` | 0.3385 +/- 0.0225 | 0.5593 +/- 0.0153 |  |  |
| `dynamic_sparse` | 0.2079 +/- 0.0341 | 0.4664 +/- 0.0201 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0292 +/- 0.0152; wins/losses/ties 2/1/0.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.1054 +/- 0.0142 | 0.2520 +/- 0.0118 |  |  |
| `recursive_features` | 0.1126 +/- 0.0211 | 0.1883 +/- 0.0015 |  |  |
| `mlp_32x32` | 0.1304 +/- 0.0314 | 0.3041 +/- 0.0318 |  |  |
| `mlp_h64` | 0.4054 +/- 0.0704 | 0.5648 +/- 0.0440 |  |  |
| `mlp_h128` | 0.5197 +/- 0.0607 | 0.6170 +/- 0.0268 |  |  |
| `mlp_h64_64` | 0.6707 +/- 0.0859 | 0.7308 +/- 0.0135 |  |  |
| `upgd_low_noise` | 0.5346 +/- 0.0877 | 0.6415 +/- 0.0270 |  |  |
| `dynamic_sparse` | 0.5630 +/- 0.1020 | 0.6788 +/- 0.0507 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0250 +/- 0.0256; wins/losses/ties 2/1/0.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0489 +/- 0.0047 | 0.0952 +/- 0.0033 |  |  |
| `recursive_features` | 0.1516 +/- 0.0194 | 0.2057 +/- 0.0066 |  |  |
| `mlp_32x32` | 0.0502 +/- 0.0055 | 0.0956 +/- 0.0051 |  |  |
| `mlp_h64` | 0.0569 +/- 0.0004 | 0.0914 +/- 0.0042 |  |  |
| `mlp_h128` | 0.0770 +/- 0.0046 | 0.1048 +/- 0.0029 |  |  |
| `mlp_h64_64` | 0.0985 +/- 0.0039 | 0.1236 +/- 0.0042 |  |  |
| `upgd_low_noise` | 0.0734 +/- 0.0049 | 0.1082 +/- 0.0032 |  |  |
| `dynamic_sparse` | 0.0838 +/- 0.0017 | 0.1153 +/- 0.0021 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0008 +/- 0.0020; wins/losses/ties 1/1/1.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.5864 +/- 0.2098 | 0.5456 +/- 0.0990 |  |  |
| `recursive_features` | 0.9916 +/- 0.3458 | 0.6653 +/- 0.1980 |  |  |
| `mlp_32x32` | 0.5801 +/- 0.2138 | 0.7326 +/- 0.0686 |  |  |
| `mlp_h64` | 0.8930 +/- 0.2378 | 1.0082 +/- 0.0855 |  |  |
| `mlp_h128` | 1.0160 +/- 0.2767 | 1.0810 +/- 0.1112 |  |  |
| `mlp_h64_64` | 1.1736 +/- 0.2645 | 1.1626 +/- 0.1066 |  |  |
| `upgd_low_noise` | 1.1864 +/- 0.2769 | 1.1972 +/- 0.1098 |  |  |
| `dynamic_sparse` | 1.2255 +/- 0.2632 | 1.1995 +/- 0.1063 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0064 +/- 0.0064; wins/losses/ties 0/1/2.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0340 +/- 0.0098 | 0.0816 +/- 0.0093 |  |  |
| `recursive_features` | 0.0854 +/- 0.0067 | 0.1142 +/- 0.0065 |  |  |
| `mlp_32x32` | 0.0330 +/- 0.0089 | 0.0776 +/- 0.0077 |  |  |
| `mlp_h64` | 0.0542 +/- 0.0088 | 0.0985 +/- 0.0090 |  |  |
| `mlp_h128` | 0.0554 +/- 0.0096 | 0.1017 +/- 0.0089 |  |  |
| `mlp_h64_64` | 0.0708 +/- 0.0097 | 0.1185 +/- 0.0092 |  |  |
| `upgd_low_noise` | 0.0580 +/- 0.0087 | 0.1052 +/- 0.0104 |  |  |
| `dynamic_sparse` | 0.0597 +/- 0.0080 | 0.1057 +/- 0.0103 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: -0.0010 +/- 0.0010; wins/losses/ties 0/1/2.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.1044 +/- 0.0302 | 0.1824 +/- 0.0136 |  |  |
| `recursive_features` | 0.1044 +/- 0.0302 | 0.1102 +/- 0.0157 |  |  |
| `mlp_32x32` | 0.6941 +/- 0.1748 | 0.8236 +/- 0.0742 |  |  |
| `mlp_h64` | 1.2230 +/- 0.2053 | 1.1382 +/- 0.0990 |  |  |
| `mlp_h128` | 1.2287 +/- 0.2042 | 1.1494 +/- 0.0959 |  |  |
| `mlp_h64_64` | 0.9397 +/- 0.2002 | 0.9356 +/- 0.0861 |  |  |
| `upgd_low_noise` | 1.2091 +/- 0.2037 | 1.1135 +/- 0.0976 |  |  |
| `dynamic_sparse` | 1.2214 +/- 0.2039 | 1.1200 +/- 0.0967 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.5897 +/- 0.1461; wins/losses/ties 3/0/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 4/6 configured datasets.

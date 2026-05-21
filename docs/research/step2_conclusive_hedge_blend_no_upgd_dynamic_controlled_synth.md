# Step 2 Conclusive Learner Candidate

Protocol: 3 seeds, 1200 steps, final window 300; route loss decay=0.99, warmup=250, guard margin=0.0.

Positive conclusive-vs-best-MLP differences favor the conclusive learner. For MSE this is best MLP minus conclusive; for accuracy this is conclusive minus best MLP.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0113 +/- 0.0012 | 0.1506 +/- 0.0083 |  |  |
| `recursive_features` | 0.1186 +/- 0.0657 | 0.1944 +/- 0.0158 |  |  |
| `polynomial_features` | 0.0113 +/- 0.0012 | 0.1732 +/- 0.0112 |  |  |
| `fourier_features` | 0.0250 +/- 0.0003 | 0.1364 +/- 0.0003 |  |  |
| `tanh_random_features` | 0.5619 +/- 0.0108 | 0.5944 +/- 0.0090 |  |  |
| `mlp_32x32_s01_no_ln` | 0.3798 +/- 0.0564 | 0.4869 +/- 0.0170 |  |  |
| `mlp_64x64_s01_no_ln` | 0.3824 +/- 0.0147 | 0.4826 +/- 0.0100 |  |  |
| `mlp_32x32` | 0.4672 +/- 0.0066 | 0.4945 +/- 0.0051 |  |  |
| `mlp_h64` | 0.3714 +/- 0.0187 | 0.5529 +/- 0.0167 |  |  |
| `mlp_h128` | 0.3599 +/- 0.0285 | 0.5559 +/- 0.0232 |  |  |
| `mlp_h64_64` | 0.1872 +/- 0.0196 | 0.4260 +/- 0.0293 |  |  |
| `upgd_low_noise` | 0.5986 +/- 0.0451 | 0.6778 +/- 0.0191 |  |  |
| `dynamic_sparse` | 0.5874 +/- 0.0131 | 0.6808 +/- 0.0140 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.1759 +/- 0.0206; wins/losses/ties 3/0/0.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0066 +/- 0.0004 | 0.2617 +/- 0.0243 |  |  |
| `recursive_features` | 0.0896 +/- 0.0300 | 0.2407 +/- 0.0387 |  |  |
| `polynomial_features` | 0.0066 +/- 0.0004 | 0.1374 +/- 0.0134 |  |  |
| `fourier_features` | 1.3656 +/- 0.0930 | 1.4496 +/- 0.0906 |  |  |
| `tanh_random_features` | 0.3801 +/- 0.0435 | 0.7439 +/- 0.0585 |  |  |
| `mlp_32x32_s01_no_ln` | 0.1524 +/- 0.0116 | 0.3510 +/- 0.0325 |  |  |
| `mlp_64x64_s01_no_ln` | 0.1517 +/- 0.0062 | 0.3665 +/- 0.0274 |  |  |
| `mlp_32x32` | 0.1438 +/- 0.0038 | 0.4355 +/- 0.0310 |  |  |
| `mlp_h64` | 0.4259 +/- 0.0269 | 0.6416 +/- 0.0605 |  |  |
| `mlp_h128` | 0.4837 +/- 0.0290 | 0.6483 +/- 0.0513 |  |  |
| `mlp_h64_64` | 0.5862 +/- 0.0524 | 0.7209 +/- 0.0599 |  |  |
| `upgd_low_noise` | 0.4587 +/- 0.0402 | 0.6800 +/- 0.0847 |  |  |
| `dynamic_sparse` | 0.4900 +/- 0.0382 | 0.7690 +/- 0.0339 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.1360 +/- 0.0053; wins/losses/ties 3/0/0.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0325 +/- 0.0038 | 0.1092 +/- 0.0080 |  |  |
| `recursive_features` | 0.2519 +/- 0.0416 | 0.3089 +/- 0.0669 |  |  |
| `polynomial_features` | 0.0537 +/- 0.0035 | 0.1462 +/- 0.0166 |  |  |
| `fourier_features` | 0.0564 +/- 0.0065 | 0.0940 +/- 0.0017 |  |  |
| `tanh_random_features` | 0.0605 +/- 0.0031 | 0.1222 +/- 0.0034 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0523 +/- 0.0065 | 0.1181 +/- 0.0096 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0489 +/- 0.0047 | 0.0996 +/- 0.0061 |  |  |
| `mlp_32x32` | 0.0436 +/- 0.0030 | 0.1204 +/- 0.0061 |  |  |
| `mlp_h64` | 0.0591 +/- 0.0070 | 0.1095 +/- 0.0008 |  |  |
| `mlp_h128` | 0.0754 +/- 0.0080 | 0.1188 +/- 0.0031 |  |  |
| `mlp_h64_64` | 0.0848 +/- 0.0067 | 0.1358 +/- 0.0061 |  |  |
| `upgd_low_noise` | 0.0735 +/- 0.0084 | 0.1232 +/- 0.0005 |  |  |
| `dynamic_sparse` | 0.0904 +/- 0.0073 | 0.1421 +/- 0.0038 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0106 +/- 0.0024; wins/losses/ties 3/0/0.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0200 +/- 0.0082 | 0.2422 +/- 0.0148 |  |  |
| `recursive_features` | 0.3263 +/- 0.1158 | 0.3517 +/- 0.0666 |  |  |
| `polynomial_features` | 0.0200 +/- 0.0082 | 0.1015 +/- 0.0121 |  |  |
| `fourier_features` | 0.5563 +/- 0.1004 | 0.7040 +/- 0.0538 |  |  |
| `tanh_random_features` | 0.7244 +/- 0.1697 | 0.9202 +/- 0.0829 |  |  |
| `mlp_32x32_s01_no_ln` | 0.5709 +/- 0.1162 | 0.7467 +/- 0.0477 |  |  |
| `mlp_64x64_s01_no_ln` | 0.5777 +/- 0.1354 | 0.7547 +/- 0.0691 |  |  |
| `mlp_32x32` | 0.5896 +/- 0.1049 | 0.7645 +/- 0.0554 |  |  |
| `mlp_h64` | 0.8025 +/- 0.1915 | 0.9548 +/- 0.0808 |  |  |
| `mlp_h128` | 0.9418 +/- 0.2211 | 1.0286 +/- 0.0855 |  |  |
| `mlp_h64_64` | 0.9750 +/- 0.2207 | 1.0838 +/- 0.1138 |  |  |
| `upgd_low_noise` | 1.0271 +/- 0.2160 | 1.0863 +/- 0.1048 |  |  |
| `dynamic_sparse` | 1.0531 +/- 0.2378 | 1.0921 +/- 0.1030 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.5304 +/- 0.1155; wins/losses/ties 3/0/0.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0345 +/- 0.0091 | 0.0917 +/- 0.0212 |  |  |
| `recursive_features` | 0.0925 +/- 0.0189 | 0.1412 +/- 0.0167 |  |  |
| `polynomial_features` | 0.0272 +/- 0.0067 | 0.0840 +/- 0.0130 |  |  |
| `fourier_features` | 0.0516 +/- 0.0150 | 0.0849 +/- 0.0215 |  |  |
| `tanh_random_features` | 0.0561 +/- 0.0120 | 0.0841 +/- 0.0218 |  |  |
| `mlp_32x32_s01_no_ln` | 0.0469 +/- 0.0114 | 0.0800 +/- 0.0193 |  |  |
| `mlp_64x64_s01_no_ln` | 0.0453 +/- 0.0111 | 0.0782 +/- 0.0183 |  |  |
| `mlp_32x32` | 0.0449 +/- 0.0122 | 0.0906 +/- 0.0215 |  |  |
| `mlp_h64` | 0.0748 +/- 0.0140 | 0.1145 +/- 0.0180 |  |  |
| `mlp_h128` | 0.0772 +/- 0.0106 | 0.1150 +/- 0.0162 |  |  |
| `mlp_h64_64` | 0.0927 +/- 0.0085 | 0.1366 +/- 0.0170 |  |  |
| `upgd_low_noise` | 0.0780 +/- 0.0090 | 0.1207 +/- 0.0197 |  |  |
| `dynamic_sparse` | 0.0778 +/- 0.0104 | 0.1212 +/- 0.0194 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0100 +/- 0.0038; wins/losses/ties 3/0/0.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.0087 +/- 0.0006 | 0.2139 +/- 0.0267 |  |  |
| `recursive_features` | 0.0796 +/- 0.0216 | 0.1309 +/- 0.0231 |  |  |
| `polynomial_features` | 0.0087 +/- 0.0006 | 0.1603 +/- 0.0162 |  |  |
| `fourier_features` | 0.8791 +/- 0.0625 | 1.1146 +/- 0.0811 |  |  |
| `tanh_random_features` | 0.5196 +/- 0.0574 | 0.9035 +/- 0.1075 |  |  |
| `mlp_32x32_s01_no_ln` | 0.4957 +/- 0.0446 | 0.8323 +/- 0.0787 |  |  |
| `mlp_64x64_s01_no_ln` | 0.4856 +/- 0.0117 | 0.8242 +/- 0.0594 |  |  |
| `mlp_32x32` | 0.6395 +/- 0.0573 | 0.9204 +/- 0.0731 |  |  |
| `mlp_h64` | 0.8523 +/- 0.0506 | 1.0847 +/- 0.0766 |  |  |
| `mlp_h128` | 0.8701 +/- 0.0501 | 1.0954 +/- 0.0781 |  |  |
| `mlp_h64_64` | 0.5883 +/- 0.0456 | 0.9108 +/- 0.0793 |  |  |
| `upgd_low_noise` | 0.8291 +/- 0.0493 | 1.0441 +/- 0.0797 |  |  |
| `dynamic_sparse` | 0.8383 +/- 0.0530 | 1.0528 +/- 0.0788 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.4601 +/- 0.0273; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.2246 +/- 0.0716 | 0.1744 +/- 0.0251 |  |  |
| `recursive_features` | 0.9415 +/- 0.3487 | 0.6405 +/- 0.1299 |  |  |
| `polynomial_features` | 1.5724 +/- 0.6786 | 0.9813 +/- 0.1784 |  |  |
| `fourier_features` | 0.4132 +/- 0.1248 | 0.2622 +/- 0.0415 |  |  |
| `tanh_random_features` | 0.2259 +/- 0.0705 | 0.1502 +/- 0.0232 |  |  |
| `mlp_32x32_s01_no_ln` | 0.4810 +/- 0.2083 | 0.3090 +/- 0.0690 |  |  |
| `mlp_64x64_s01_no_ln` | 0.5581 +/- 0.2385 | 0.3283 +/- 0.0714 |  |  |
| `mlp_32x32` | 0.4908 +/- 0.2024 | 0.3256 +/- 0.0680 |  |  |
| `mlp_h64` | 0.2758 +/- 0.0955 | 0.1798 +/- 0.0286 |  |  |
| `mlp_h128` | 0.2757 +/- 0.0919 | 0.1799 +/- 0.0299 |  |  |
| `mlp_h64_64` | 0.3683 +/- 0.1354 | 0.2323 +/- 0.0404 |  |  |
| `upgd_low_noise` | 0.4463 +/- 0.1682 | 0.2716 +/- 0.0554 |  |  |
| `dynamic_sparse` | 0.4086 +/- 0.1523 | 0.2563 +/- 0.0493 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.0481 +/- 0.0217; wins/losses/ties 3/0/0.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.6147 +/- 0.0743 | 0.7988 +/- 0.2000 |  |  |
| `recursive_features` | 2.6877 +/- 0.3196 | 2.3972 +/- 0.2927 |  |  |
| `polynomial_features` | 3.6461 +/- 0.3127 | 3.7931 +/- 0.8853 |  |  |
| `fourier_features` | 0.6017 +/- 0.0715 | 0.5938 +/- 0.1313 |  |  |
| `tanh_random_features` | 1.1619 +/- 0.2714 | 1.3958 +/- 0.2891 |  |  |
| `mlp_32x32_s01_no_ln` | 1.0711 +/- 0.2362 | 1.2792 +/- 0.2465 |  |  |
| `mlp_64x64_s01_no_ln` | 1.1322 +/- 0.1744 | 1.2777 +/- 0.2471 |  |  |
| `mlp_32x32` | 1.0478 +/- 0.2185 | 1.2173 +/- 0.2313 |  |  |
| `mlp_h64` | 1.1387 +/- 0.2732 | 1.3298 +/- 0.2423 |  |  |
| `mlp_h128` | 1.1748 +/- 0.2556 | 1.3506 +/- 0.2485 |  |  |
| `mlp_h64_64` | 1.1524 +/- 0.2666 | 1.3418 +/- 0.2434 |  |  |
| `upgd_low_noise` | 1.2374 +/- 0.1809 | 1.3488 +/- 0.2702 |  |  |
| `dynamic_sparse` | 1.1872 +/- 0.2036 | 1.3313 +/- 0.2693 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.4230 +/- 0.2131; wins/losses/ties 3/0/0.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `conclusive` | 0.7756 +/- 0.4899 | 0.8887 +/- 0.2145 |  |  |
| `recursive_features` | 0.9570 +/- 0.6651 | 0.9717 +/- 0.2227 |  |  |
| `polynomial_features` | 0.8507 +/- 0.3960 | 0.8710 +/- 0.2091 |  |  |
| `fourier_features` | 1.0094 +/- 0.6904 | 1.0752 +/- 0.2645 |  |  |
| `tanh_random_features` | 1.0354 +/- 0.7231 | 1.0927 +/- 0.2605 |  |  |
| `mlp_32x32_s01_no_ln` | 0.9170 +/- 0.6354 | 0.9781 +/- 0.2300 |  |  |
| `mlp_64x64_s01_no_ln` | 0.9064 +/- 0.6316 | 0.9593 +/- 0.2281 |  |  |
| `mlp_32x32` | 0.9142 +/- 0.6476 | 0.9711 +/- 0.2329 |  |  |
| `mlp_h64` | 0.9662 +/- 0.6492 | 1.0431 +/- 0.2395 |  |  |
| `mlp_h128` | 0.9714 +/- 0.6512 | 1.0442 +/- 0.2380 |  |  |
| `mlp_h64_64` | 0.9456 +/- 0.6441 | 1.0231 +/- 0.2348 |  |  |
| `upgd_low_noise` | 0.9184 +/- 0.6359 | 0.9763 +/- 0.2242 |  |  |
| `dynamic_sparse` | 0.9175 +/- 0.6354 | 0.9826 +/- 0.2303 |  |  |

`final_window_mse` conclusive-vs-best-MLP diff: +0.1254 +/- 0.1445; wins/losses/ties 1/2/0.

## Suite Summary

Conclusive learner has positive mean final-window MSE delta against best fair MLP on 9/9 configured datasets.

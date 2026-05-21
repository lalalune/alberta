# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 0.1121 +/- 0.0062 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 0.1280 +/- 0.0035 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 0.1357 +/- 0.0044 |
| `d18_step2_canonical` | 0.0418 +/- 0.0015 | 0.1198 +/- 0.0011 |  |  | 320.0000 +/- 0.0000 | 1.1496 +/- 0.0041 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1150 +/- 0.0276; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.1238 +/- 0.0141 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.1405 +/- 0.0365 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 0.1630 +/- 0.0161 |
| `d18_step2_canonical` | 0.0296 +/- 0.0036 | 0.1050 +/- 0.0114 |  |  | 320.0000 +/- 0.0000 | 1.2145 +/- 0.0513 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3986 +/- 0.0214; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0580 +/- 0.0034 | 0.1073 +/- 0.0013 |  |  |  | 0.3763 +/- 0.2554 |
| `mlp_h128` | 0.0751 +/- 0.0086 | 0.1187 +/- 0.0048 |  |  |  | 0.3835 +/- 0.2469 |
| `mlp_h64_64` | 0.0750 +/- 0.0060 | 0.1274 +/- 0.0034 |  |  |  | 0.3002 +/- 0.1196 |
| `d18_step2_canonical` | 0.0172 +/- 0.0012 | 0.0439 +/- 0.0005 |  |  | 320.0000 +/- 0.0000 | 1.4432 +/- 0.1032 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0408 +/- 0.0032; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8611 +/- 0.1893 | 0.9917 +/- 0.0835 |  |  |  | 0.1122 +/- 0.0033 |
| `mlp_h128` | 0.9468 +/- 0.2304 | 1.0382 +/- 0.0979 |  |  |  | 0.1069 +/- 0.0025 |
| `mlp_h64_64` | 0.9490 +/- 0.2284 | 1.0421 +/- 0.1050 |  |  |  | 0.1507 +/- 0.0059 |
| `d18_step2_canonical` | 0.0511 +/- 0.0160 | 0.2433 +/- 0.0312 |  |  | 320.0000 +/- 0.0000 | 1.1560 +/- 0.0075 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.8088 +/- 0.1741; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.1303 +/- 0.0047 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.1366 +/- 0.0021 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.1767 +/- 0.0055 |
| `d18_step2_canonical` | 0.0195 +/- 0.0066 | 0.0529 +/- 0.0123 |  |  | 320.0000 +/- 0.0000 | 1.1811 +/- 0.0083 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0537 +/- 0.0055; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8637 +/- 0.0489 | 1.0904 +/- 0.0786 |  |  |  | 0.1033 +/- 0.0029 |
| `mlp_h128` | 0.8693 +/- 0.0485 | 1.0951 +/- 0.0761 |  |  |  | 0.1044 +/- 0.0047 |
| `mlp_h64_64` | 0.6059 +/- 0.0368 | 0.9161 +/- 0.0726 |  |  |  | 0.1576 +/- 0.0085 |
| `d18_step2_canonical` | 0.0431 +/- 0.0059 | 0.2494 +/- 0.0176 |  |  | 320.0000 +/- 0.0000 | 1.1544 +/- 0.0111 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5628 +/- 0.0315; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 0.3635 +/- 0.0280 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 0.3610 +/- 0.0274 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 0.4079 +/- 0.0170 |
| `d18_step2_canonical` | 0.0018 +/- 0.0004 | 0.0020 +/- 0.0002 | 0.9911 +/- 0.0022 | 0.8602 +/- 0.0146 | 320.0000 +/- 0.0000 | 2.0992 +/- 0.1168 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0012 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.6945 +/- 0.0305; wins/losses/ties 3/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0310 +/- 0.0011 | 0.0440 +/- 0.0001 | 0.9133 +/- 0.0084 | 0.9264 +/- 0.0070 |  | 0.3741 +/- 0.0136 |
| `mlp_h128` | 0.0314 +/- 0.0010 | 0.0438 +/- 0.0003 | 0.9089 +/- 0.0056 | 0.9338 +/- 0.0016 |  | 0.5285 +/- 0.1361 |
| `mlp_h64_64` | 0.0332 +/- 0.0017 | 0.0478 +/- 0.0012 | 0.8789 +/- 0.0146 | 0.9109 +/- 0.0057 |  | 0.5171 +/- 0.0315 |
| `d18_step2_canonical` | 0.0207 +/- 0.0007 | 0.0298 +/- 0.0004 | 0.9644 +/- 0.0029 | 0.9654 +/- 0.0031 | 320.0000 +/- 0.0000 | 1.9240 +/- 0.0642 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0102 +/- 0.0006; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0309 +/- 0.0022; wins/losses/ties 3/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 0.2892 +/- 0.0059 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 0.3027 +/- 0.0068 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 0.3308 +/- 0.0056 |
| `d18_step2_canonical` | 0.0334 +/- 0.0006 | 0.0458 +/- 0.0005 | 0.8944 +/- 0.0106 | 0.9518 +/- 0.0067 | 320.0000 +/- 0.0000 | 1.8479 +/- 0.0046 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0047 +/- 0.0012; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0445 +/- 0.0074; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.3074 +/- 0.0127 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 0.3910 +/- 0.0314 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.3970 +/- 0.0359 |
| `d18_step2_canonical` | 0.0458 +/- 0.0013 | 0.0537 +/- 0.0013 | 0.7789 +/- 0.0198 | 0.8281 +/- 0.0220 | 320.0000 +/- 0.0000 | 2.0972 +/- 0.0546 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0020 +/- 0.0008; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0080 +/- 0.0107; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 0.3802 +/- 0.0472 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 0.4244 +/- 0.0414 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 0.5154 +/- 0.0389 |
| `d18_step2_canonical` | 0.0381 +/- 0.0013 | 0.0461 +/- 0.0002 | 0.8744 +/- 0.0091 | 0.9221 +/- 0.0103 | 320.0000 +/- 0.0000 | 2.0153 +/- 0.0232 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0110 +/- 0.0014; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0408 +/- 0.0021; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.4819 +/- 1.3003 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.4106 +/- 0.1911 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.2768 +/- 0.0199 |
| `d18_step2_canonical` | 0.2310 +/- 0.0786 | 0.1464 +/- 0.0260 |  |  | 320.0000 +/- 0.0000 | 1.3891 +/- 0.0134 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0407 +/- 0.0168; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.1599 +/- 0.0154 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.1575 +/- 0.0141 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.2059 +/- 0.0080 |
| `d18_step2_canonical` | 0.8397 +/- 0.2223 | 0.9428 +/- 0.1908 |  |  | 320.0000 +/- 0.0000 | 1.2891 +/- 0.0478 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3088 +/- 0.0491; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.3187 +/- 0.1528 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.2901 +/- 0.1421 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.1897 +/- 0.0049 |
| `d18_step2_canonical` | 0.8190 +/- 0.5361 | 0.8371 +/- 0.2138 |  |  | 320.0000 +/- 0.0000 | 1.3999 +/- 0.0046 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1285 +/- 0.1083; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 0.1042 +/- 0.0011 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 0.1100 +/- 0.0036 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 0.1570 +/- 0.0037 |
| `d18_step2_canonical` | 0.0395 +/- 0.0016 | 0.1112 +/- 0.0007 |  |  | 320.0000 +/- 0.0000 | 1.1439 +/- 0.0048 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1173 +/- 0.0273; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.1422 +/- 0.0268 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.1203 +/- 0.0114 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 0.1687 +/- 0.0112 |
| `d18_step2_canonical` | 0.0296 +/- 0.0036 | 0.1053 +/- 0.0110 |  |  | 320.0000 +/- 0.0000 | 1.1862 +/- 0.0396 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3987 +/- 0.0216; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0580 +/- 0.0034 | 0.1073 +/- 0.0013 |  |  |  | 0.3980 +/- 0.2588 |
| `mlp_h128` | 0.0751 +/- 0.0086 | 0.1187 +/- 0.0048 |  |  |  | 0.3958 +/- 0.2627 |
| `mlp_h64_64` | 0.0750 +/- 0.0060 | 0.1274 +/- 0.0034 |  |  |  | 0.3104 +/- 0.1139 |
| `d18_step2_canonical` | 0.0172 +/- 0.0010 | 0.0440 +/- 0.0001 |  |  | 320.0000 +/- 0.0000 | 1.4314 +/- 0.1028 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0408 +/- 0.0033; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8611 +/- 0.1893 | 0.9917 +/- 0.0835 |  |  |  | 0.1035 +/- 0.0018 |
| `mlp_h128` | 0.9468 +/- 0.2304 | 1.0382 +/- 0.0979 |  |  |  | 0.1113 +/- 0.0065 |
| `mlp_h64_64` | 0.9490 +/- 0.2284 | 1.0421 +/- 0.1050 |  |  |  | 0.1419 +/- 0.0035 |
| `d18_step2_canonical` | 0.0530 +/- 0.0180 | 0.2475 +/- 0.0337 |  |  | 320.0000 +/- 0.0000 | 1.1379 +/- 0.0007 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.8070 +/- 0.1720; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.1313 +/- 0.0060 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.1352 +/- 0.0072 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.1853 +/- 0.0052 |
| `d18_step2_canonical` | 0.0196 +/- 0.0067 | 0.0530 +/- 0.0122 |  |  | 320.0000 +/- 0.0000 | 1.1522 +/- 0.0058 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0536 +/- 0.0054; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8637 +/- 0.0489 | 1.0904 +/- 0.0786 |  |  |  | 0.1086 +/- 0.0034 |
| `mlp_h128` | 0.8693 +/- 0.0485 | 1.0951 +/- 0.0761 |  |  |  | 0.1140 +/- 0.0030 |
| `mlp_h64_64` | 0.6059 +/- 0.0368 | 0.9161 +/- 0.0726 |  |  |  | 0.1665 +/- 0.0091 |
| `d18_step2_canonical` | 0.0438 +/- 0.0056 | 0.2488 +/- 0.0168 |  |  | 320.0000 +/- 0.0000 | 1.1366 +/- 0.0040 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5621 +/- 0.0318; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 0.3582 +/- 0.0252 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 0.3806 +/- 0.0108 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 0.4526 +/- 0.0223 |
| `d18_step2_canonical` | 0.0016 +/- 0.0002 | 0.0020 +/- 0.0002 | 0.9922 +/- 0.0011 | 0.8609 +/- 0.0159 | 320.0000 +/- 0.0000 | 2.1028 +/- 0.0914 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0014 +/- 0.0003; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.6951 +/- 0.0316; wins/losses/ties 3/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0310 +/- 0.0011 | 0.0440 +/- 0.0001 | 0.9133 +/- 0.0084 | 0.9264 +/- 0.0070 |  | 0.3720 +/- 0.0205 |
| `mlp_h128` | 0.0314 +/- 0.0010 | 0.0438 +/- 0.0003 | 0.9089 +/- 0.0056 | 0.9338 +/- 0.0016 |  | 0.5293 +/- 0.1209 |
| `mlp_h64_64` | 0.0332 +/- 0.0017 | 0.0478 +/- 0.0012 | 0.8789 +/- 0.0146 | 0.9109 +/- 0.0057 |  | 0.5069 +/- 0.0495 |
| `d18_step2_canonical` | 0.0210 +/- 0.0008 | 0.0300 +/- 0.0004 | 0.9622 +/- 0.0029 | 0.9660 +/- 0.0025 | 320.0000 +/- 0.0000 | 1.9870 +/- 0.0672 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0099 +/- 0.0006; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0315 +/- 0.0019; wins/losses/ties 3/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 0.2950 +/- 0.0037 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 0.3022 +/- 0.0070 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 0.3308 +/- 0.0069 |
| `d18_step2_canonical` | 0.0338 +/- 0.0005 | 0.0460 +/- 0.0007 | 0.8956 +/- 0.0099 | 0.9493 +/- 0.0089 | 320.0000 +/- 0.0000 | 1.9206 +/- 0.0207 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0043 +/- 0.0010; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0421 +/- 0.0096; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.3552 +/- 0.0181 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 0.5745 +/- 0.1582 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.3933 +/- 0.0201 |
| `d18_step2_canonical` | 0.0457 +/- 0.0013 | 0.0538 +/- 0.0013 | 0.7844 +/- 0.0251 | 0.8312 +/- 0.0204 | 320.0000 +/- 0.0000 | 1.9686 +/- 0.0513 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0021 +/- 0.0006; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0111 +/- 0.0098; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 0.4679 +/- 0.0505 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 0.4517 +/- 0.0327 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 0.4532 +/- 0.0345 |
| `d18_step2_canonical` | 0.0383 +/- 0.0013 | 0.0462 +/- 0.0002 | 0.8800 +/- 0.0120 | 0.9233 +/- 0.0108 | 320.0000 +/- 0.0000 | 2.0158 +/- 0.0150 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0108 +/- 0.0012; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0421 +/- 0.0034; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.4475 +/- 1.2695 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.4314 +/- 0.2006 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.2624 +/- 0.0146 |
| `d18_step2_canonical` | 0.2367 +/- 0.0816 | 0.1485 +/- 0.0261 |  |  | 320.0000 +/- 0.0000 | 1.3892 +/- 0.0298 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0350 +/- 0.0138; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.1417 +/- 0.0112 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.1676 +/- 0.0044 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.2097 +/- 0.0120 |
| `d18_step2_canonical` | 0.8379 +/- 0.2104 | 0.9473 +/- 0.1884 |  |  | 320.0000 +/- 0.0000 | 1.3370 +/- 0.0634 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3106 +/- 0.0585; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.3159 +/- 0.1528 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.2951 +/- 0.1344 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.1934 +/- 0.0042 |
| `d18_step2_canonical` | 0.8212 +/- 0.5384 | 0.8359 +/- 0.2122 |  |  | 320.0000 +/- 0.0000 | 1.4126 +/- 0.0187 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1263 +/- 0.1061; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

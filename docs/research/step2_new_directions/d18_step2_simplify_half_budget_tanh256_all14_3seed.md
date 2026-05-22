# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 0.2559 +/- 0.0393 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 0.2876 +/- 0.0291 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 0.3558 +/- 0.0452 |
| `d18_step2_canonical` | 0.0598 +/- 0.0028 | 0.1272 +/- 0.0029 |  |  | 160.0000 +/- 0.0000 | 2.4506 +/- 0.1305 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0970 +/- 0.0243; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.3462 +/- 0.1010 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.2619 +/- 0.0253 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 0.4337 +/- 0.1010 |
| `d18_step2_canonical` | 0.0370 +/- 0.0040 | 0.1318 +/- 0.0163 |  |  | 160.0000 +/- 0.0000 | 2.7550 +/- 0.0793 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3913 +/- 0.0204; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0580 +/- 0.0034 | 0.1073 +/- 0.0013 |  |  |  | 0.7156 +/- 0.4879 |
| `mlp_h128` | 0.0751 +/- 0.0086 | 0.1187 +/- 0.0048 |  |  |  | 0.6925 +/- 0.4186 |
| `mlp_h64_64` | 0.0750 +/- 0.0060 | 0.1274 +/- 0.0034 |  |  |  | 0.5307 +/- 0.2097 |
| `d18_step2_canonical` | 0.0160 +/- 0.0007 | 0.0462 +/- 0.0013 |  |  | 160.0000 +/- 0.0000 | 2.7768 +/- 0.0519 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0419 +/- 0.0028; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8611 +/- 0.1893 | 0.9917 +/- 0.0835 |  |  |  | 0.2236 +/- 0.0290 |
| `mlp_h128` | 0.9468 +/- 0.2304 | 1.0382 +/- 0.0979 |  |  |  | 0.2692 +/- 0.0344 |
| `mlp_h64_64` | 0.9490 +/- 0.2284 | 1.0421 +/- 0.1050 |  |  |  | 0.2620 +/- 0.0381 |
| `d18_step2_canonical` | 0.1161 +/- 0.0383 | 0.3115 +/- 0.0333 |  |  | 160.0000 +/- 0.0000 | 2.4097 +/- 0.1675 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.7438 +/- 0.1538; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.3969 +/- 0.0717 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.2902 +/- 0.0218 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.5586 +/- 0.0996 |
| `d18_step2_canonical` | 0.0180 +/- 0.0051 | 0.0534 +/- 0.0133 |  |  | 160.0000 +/- 0.0000 | 3.1015 +/- 0.2819 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0551 +/- 0.0065; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8637 +/- 0.0489 | 1.0904 +/- 0.0786 |  |  |  | 0.2769 +/- 0.0586 |
| `mlp_h128` | 0.8693 +/- 0.0485 | 1.0951 +/- 0.0761 |  |  |  | 0.2480 +/- 0.0081 |
| `mlp_h64_64` | 0.6059 +/- 0.0368 | 0.9161 +/- 0.0726 |  |  |  | 0.3751 +/- 0.0137 |
| `d18_step2_canonical` | 0.0933 +/- 0.0072 | 0.3355 +/- 0.0340 |  |  | 160.0000 +/- 0.0000 | 2.4962 +/- 0.0856 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5126 +/- 0.0428; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 0.7231 +/- 0.0468 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 0.6870 +/- 0.0222 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 0.7459 +/- 0.0898 |
| `d18_step2_canonical` | 0.0020 +/- 0.0004 | 0.0022 +/- 0.0002 | 0.9900 +/- 0.0019 | 0.8615 +/- 0.0164 | 160.0000 +/- 0.0000 | 4.2003 +/- 0.5455 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0010 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.6957 +/- 0.0311; wins/losses/ties 3/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0310 +/- 0.0011 | 0.0440 +/- 0.0001 | 0.9133 +/- 0.0084 | 0.9264 +/- 0.0070 |  | 0.6020 +/- 0.0167 |
| `mlp_h128` | 0.0314 +/- 0.0010 | 0.0438 +/- 0.0003 | 0.9089 +/- 0.0056 | 0.9338 +/- 0.0016 |  | 0.8463 +/- 0.1652 |
| `mlp_h64_64` | 0.0332 +/- 0.0017 | 0.0478 +/- 0.0012 | 0.8789 +/- 0.0146 | 0.9109 +/- 0.0057 |  | 0.7556 +/- 0.0993 |
| `d18_step2_canonical` | 0.0232 +/- 0.0008 | 0.0319 +/- 0.0003 | 0.9478 +/- 0.0029 | 0.9586 +/- 0.0012 | 160.0000 +/- 0.0000 | 3.1464 +/- 0.1360 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0078 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0241 +/- 0.0011; wins/losses/ties 3/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 0.8927 +/- 0.1426 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 0.9302 +/- 0.0936 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 1.0176 +/- 0.0659 |
| `d18_step2_canonical` | 0.0351 +/- 0.0005 | 0.0470 +/- 0.0006 | 0.8800 +/- 0.0102 | 0.9400 +/- 0.0027 | 160.0000 +/- 0.0000 | 5.8259 +/- 0.6677 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0030 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0328 +/- 0.0031; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.7034 +/- 0.0713 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 1.0598 +/- 0.1746 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.8431 +/- 0.0432 |
| `d18_step2_canonical` | 0.0468 +/- 0.0018 | 0.0546 +/- 0.0011 | 0.7844 +/- 0.0260 | 0.8089 +/- 0.0193 | 160.0000 +/- 0.0000 | 6.4721 +/- 0.7374 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0011 +/- 0.0012; wins/losses/ties 2/1/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0111 +/- 0.0056; wins/losses/ties 0/3/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 0.9743 +/- 0.0609 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 0.5551 +/- 0.0240 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 0.7502 +/- 0.1238 |
| `d18_step2_canonical` | 0.0419 +/- 0.0013 | 0.0482 +/- 0.0001 | 0.8556 +/- 0.0175 | 0.9011 +/- 0.0053 | 160.0000 +/- 0.0000 | 4.3604 +/- 0.6146 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0073 +/- 0.0014; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0198 +/- 0.0071; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.5812 +/- 0.3306 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.6704 +/- 0.3030 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.3826 +/- 0.0264 |
| `d18_step2_canonical` | 0.2316 +/- 0.0826 | 0.1486 +/- 0.0269 |  |  | 160.0000 +/- 0.0000 | 2.5684 +/- 0.2088 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0401 +/- 0.0131; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.3160 +/- 0.0933 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.3415 +/- 0.0662 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.3477 +/- 0.0119 |
| `d18_step2_canonical` | 0.8197 +/- 0.2082 | 0.9235 +/- 0.1787 |  |  | 160.0000 +/- 0.0000 | 3.0762 +/- 0.1979 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3288 +/- 0.0635; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.5973 +/- 0.3552 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.6179 +/- 0.3450 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.3889 +/- 0.0685 |
| `d18_step2_canonical` | 0.8407 +/- 0.5589 | 0.8478 +/- 0.2144 |  |  | 160.0000 +/- 0.0000 | 2.6479 +/- 0.1622 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1067 +/- 0.0856; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

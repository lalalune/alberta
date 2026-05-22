# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 0.3088 +/- 0.0541 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 0.2999 +/- 0.0223 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 0.4460 +/- 0.0344 |
| `d18_step2_canonical` | 0.0393 +/- 0.0018 | 0.1053 +/- 0.0005 |  |  | 320.0000 +/- 0.0000 | 3.6836 +/- 0.7725 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1176 +/- 0.0273; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.5457 +/- 0.2182 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.4576 +/- 0.0485 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 0.5580 +/- 0.1412 |
| `d18_step2_canonical` | 0.0296 +/- 0.0036 | 0.1054 +/- 0.0110 |  |  | 320.0000 +/- 0.0000 | 10.0268 +/- 1.3670 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3987 +/- 0.0216; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0580 +/- 0.0034 | 0.1073 +/- 0.0013 |  |  |  | 0.8170 +/- 0.6186 |
| `mlp_h128` | 0.0751 +/- 0.0086 | 0.1187 +/- 0.0048 |  |  |  | 0.6945 +/- 0.4338 |
| `mlp_h64_64` | 0.0750 +/- 0.0060 | 0.1274 +/- 0.0034 |  |  |  | 0.5055 +/- 0.2085 |
| `d18_step2_canonical` | 0.0172 +/- 0.0010 | 0.0442 +/- 0.0001 |  |  | 320.0000 +/- 0.0000 | 3.1962 +/- 1.0153 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0408 +/- 0.0033; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8611 +/- 0.1893 | 0.9917 +/- 0.0835 |  |  |  | 0.2021 +/- 0.0057 |
| `mlp_h128` | 0.9468 +/- 0.2304 | 1.0382 +/- 0.0979 |  |  |  | 0.2231 +/- 0.0211 |
| `mlp_h64_64` | 0.9490 +/- 0.2284 | 1.0421 +/- 0.1050 |  |  |  | 0.3724 +/- 0.0517 |
| `d18_step2_canonical` | 0.0529 +/- 0.0183 | 0.2483 +/- 0.0340 |  |  | 320.0000 +/- 0.0000 | 4.2392 +/- 1.8436 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.8070 +/- 0.1717; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.3388 +/- 0.0682 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.3824 +/- 0.0152 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.4052 +/- 0.0394 |
| `d18_step2_canonical` | 0.0195 +/- 0.0067 | 0.0529 +/- 0.0122 |  |  | 320.0000 +/- 0.0000 | 3.7835 +/- 0.6414 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0537 +/- 0.0054; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8637 +/- 0.0489 | 1.0904 +/- 0.0786 |  |  |  | 0.5513 +/- 0.0925 |
| `mlp_h128` | 0.8693 +/- 0.0485 | 1.0951 +/- 0.0761 |  |  |  | 0.5638 +/- 0.0718 |
| `mlp_h64_64` | 0.6059 +/- 0.0368 | 0.9161 +/- 0.0726 |  |  |  | 0.6978 +/- 0.0821 |
| `d18_step2_canonical` | 0.0438 +/- 0.0056 | 0.2492 +/- 0.0172 |  |  | 320.0000 +/- 0.0000 | 9.9066 +/- 2.1341 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5621 +/- 0.0319; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 2.8287 +/- 0.8862 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 1.9731 +/- 0.8256 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 4.7485 +/- 2.0259 |
| `d18_step2_canonical` | 0.0016 +/- 0.0002 | 0.0020 +/- 0.0002 | 0.9922 +/- 0.0011 | 0.8609 +/- 0.0159 | 320.0000 +/- 0.0000 | 19.1280 +/- 3.4304 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0014 +/- 0.0003; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.6951 +/- 0.0316; wins/losses/ties 3/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0310 +/- 0.0011 | 0.0440 +/- 0.0001 | 0.9133 +/- 0.0084 | 0.9264 +/- 0.0070 |  | 1.5855 +/- 0.3461 |
| `mlp_h128` | 0.0314 +/- 0.0010 | 0.0438 +/- 0.0003 | 0.9089 +/- 0.0056 | 0.9338 +/- 0.0016 |  | 2.1444 +/- 0.7987 |
| `mlp_h64_64` | 0.0332 +/- 0.0017 | 0.0478 +/- 0.0012 | 0.8789 +/- 0.0146 | 0.9109 +/- 0.0057 |  | 2.8948 +/- 0.8954 |
| `d18_step2_canonical` | 0.0210 +/- 0.0008 | 0.0300 +/- 0.0004 | 0.9622 +/- 0.0029 | 0.9660 +/- 0.0025 | 320.0000 +/- 0.0000 | 12.5425 +/- 3.1900 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0099 +/- 0.0006; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0315 +/- 0.0019; wins/losses/ties 3/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 2.2078 +/- 0.2249 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 1.7505 +/- 0.1615 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 3.2223 +/- 0.6321 |
| `d18_step2_canonical` | 0.0338 +/- 0.0005 | 0.0460 +/- 0.0007 | 0.8956 +/- 0.0099 | 0.9499 +/- 0.0088 | 320.0000 +/- 0.0000 | 24.1997 +/- 2.2473 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0043 +/- 0.0010; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0427 +/- 0.0093; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 2.5905 +/- 0.2448 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 3.2831 +/- 0.9430 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 3.1308 +/- 0.8916 |
| `d18_step2_canonical` | 0.0458 +/- 0.0013 | 0.0539 +/- 0.0013 | 0.7833 +/- 0.0255 | 0.8336 +/- 0.0193 | 320.0000 +/- 0.0000 | 21.3230 +/- 5.2970 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0020 +/- 0.0006; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0136 +/- 0.0082; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 4.1138 +/- 1.0037 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 3.4997 +/- 0.8221 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 3.0181 +/- 0.6561 |
| `d18_step2_canonical` | 0.0383 +/- 0.0013 | 0.0462 +/- 0.0002 | 0.8822 +/- 0.0128 | 0.9239 +/- 0.0093 | 320.0000 +/- 0.0000 | 26.6685 +/- 1.7375 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0108 +/- 0.0012; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0427 +/- 0.0039; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.1569 +/- 0.4352 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.3246 +/- 0.2661 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.1305 +/- 0.5655 |
| `d18_step2_canonical` | 0.2370 +/- 0.0817 | 0.1485 +/- 0.0261 |  |  | 320.0000 +/- 0.0000 | 16.6440 +/- 6.1254 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0347 +/- 0.0137; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 1.1230 +/- 0.3757 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 1.2145 +/- 0.5036 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 1.3026 +/- 0.4074 |
| `d18_step2_canonical` | 0.8349 +/- 0.2068 | 0.9475 +/- 0.1886 |  |  | 320.0000 +/- 0.0000 | 16.9409 +/- 4.9734 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3136 +/- 0.0608; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.0617 +/- 0.4068 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.0001 +/- 0.2919 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.6868 +/- 0.1946 |
| `d18_step2_canonical` | 0.8205 +/- 0.5376 | 0.8347 +/- 0.2117 |  |  | 320.0000 +/- 0.0000 | 8.2529 +/- 2.6474 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1270 +/- 0.1068; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 0.3848 +/- 0.0768 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 0.4575 +/- 0.0486 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 0.5764 +/- 0.1221 |
| `d18_step2_canonical` | 0.0529 +/- 0.0021 | 0.1160 +/- 0.0018 |  |  | 160.0000 +/- 0.0000 | 2.3068 +/- 0.1216 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1039 +/- 0.0244; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.3154 +/- 0.0621 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.3993 +/- 0.0271 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 0.4689 +/- 0.0314 |
| `d18_step2_canonical` | 0.0369 +/- 0.0038 | 0.1314 +/- 0.0153 |  |  | 160.0000 +/- 0.0000 | 4.2657 +/- 0.2864 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3913 +/- 0.0206; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0580 +/- 0.0034 | 0.1073 +/- 0.0013 |  |  |  | 0.9929 +/- 0.7004 |
| `mlp_h128` | 0.0751 +/- 0.0086 | 0.1187 +/- 0.0048 |  |  |  | 0.7495 +/- 0.4999 |
| `mlp_h64_64` | 0.0750 +/- 0.0060 | 0.1274 +/- 0.0034 |  |  |  | 0.7799 +/- 0.1946 |
| `d18_step2_canonical` | 0.0161 +/- 0.0006 | 0.0463 +/- 0.0013 |  |  | 160.0000 +/- 0.0000 | 4.8776 +/- 0.2019 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0419 +/- 0.0030; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8611 +/- 0.1893 | 0.9917 +/- 0.0835 |  |  |  | 0.2584 +/- 0.0203 |
| `mlp_h128` | 0.9468 +/- 0.2304 | 1.0382 +/- 0.0979 |  |  |  | 0.5117 +/- 0.0680 |
| `mlp_h64_64` | 0.9490 +/- 0.2284 | 1.0421 +/- 0.1050 |  |  |  | 0.6068 +/- 0.1322 |
| `d18_step2_canonical` | 0.1178 +/- 0.0396 | 0.3125 +/- 0.0336 |  |  | 160.0000 +/- 0.0000 | 2.3230 +/- 0.2615 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.7421 +/- 0.1524; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.3313 +/- 0.0516 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.4957 +/- 0.1401 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.4752 +/- 0.0677 |
| `d18_step2_canonical` | 0.0181 +/- 0.0052 | 0.0536 +/- 0.0132 |  |  | 160.0000 +/- 0.0000 | 2.0892 +/- 0.0204 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0551 +/- 0.0065; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8637 +/- 0.0489 | 1.0904 +/- 0.0786 |  |  |  | 0.3451 +/- 0.0465 |
| `mlp_h128` | 0.8693 +/- 0.0485 | 1.0951 +/- 0.0761 |  |  |  | 0.3252 +/- 0.0181 |
| `mlp_h64_64` | 0.6059 +/- 0.0368 | 0.9161 +/- 0.0726 |  |  |  | 0.3335 +/- 0.0205 |
| `d18_step2_canonical` | 0.0931 +/- 0.0073 | 0.3339 +/- 0.0331 |  |  | 160.0000 +/- 0.0000 | 2.3295 +/- 0.2224 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5128 +/- 0.0429; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 0.9057 +/- 0.1474 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 0.8712 +/- 0.0893 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 1.0391 +/- 0.0483 |
| `d18_step2_canonical` | 0.0018 +/- 0.0002 | 0.0021 +/- 0.0001 | 0.9911 +/- 0.0011 | 0.8627 +/- 0.0170 | 160.0000 +/- 0.0000 | 4.6780 +/- 0.0646 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0012 +/- 0.0003; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.6970 +/- 0.0317; wins/losses/ties 3/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0310 +/- 0.0011 | 0.0440 +/- 0.0001 | 0.9133 +/- 0.0084 | 0.9264 +/- 0.0070 |  | 0.9178 +/- 0.0910 |
| `mlp_h128` | 0.0314 +/- 0.0010 | 0.0438 +/- 0.0003 | 0.9089 +/- 0.0056 | 0.9338 +/- 0.0016 |  | 1.2424 +/- 0.3833 |
| `mlp_h64_64` | 0.0332 +/- 0.0017 | 0.0478 +/- 0.0012 | 0.8789 +/- 0.0146 | 0.9109 +/- 0.0057 |  | 1.0403 +/- 0.1371 |
| `d18_step2_canonical` | 0.0235 +/- 0.0008 | 0.0322 +/- 0.0003 | 0.9422 +/- 0.0029 | 0.9561 +/- 0.0012 | 160.0000 +/- 0.0000 | 4.4915 +/- 0.1085 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0074 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0216 +/- 0.0016; wins/losses/ties 3/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 0.8517 +/- 0.0284 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 0.8003 +/- 0.0692 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 1.0368 +/- 0.0329 |
| `d18_step2_canonical` | 0.0356 +/- 0.0006 | 0.0472 +/- 0.0007 | 0.8756 +/- 0.0068 | 0.9363 +/- 0.0071 | 160.0000 +/- 0.0000 | 4.0398 +/- 0.0977 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0025 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0291 +/- 0.0065; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.7796 +/- 0.0197 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 0.9467 +/- 0.0689 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.7929 +/- 0.0169 |
| `d18_step2_canonical` | 0.0470 +/- 0.0018 | 0.0548 +/- 0.0012 | 0.7844 +/- 0.0225 | 0.8120 +/- 0.0237 | 160.0000 +/- 0.0000 | 4.5486 +/- 0.1385 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0009 +/- 0.0011; wins/losses/ties 2/1/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0080 +/- 0.0108; wins/losses/ties 2/1/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 0.8585 +/- 0.1512 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 0.8136 +/- 0.0366 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 1.0332 +/- 0.0430 |
| `d18_step2_canonical` | 0.0421 +/- 0.0014 | 0.0484 +/- 0.0001 | 0.8567 +/- 0.0190 | 0.8961 +/- 0.0088 | 160.0000 +/- 0.0000 | 4.2361 +/- 0.4472 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0071 +/- 0.0013; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0148 +/- 0.0065; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.1303 +/- 0.6887 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.9071 +/- 0.4813 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.5454 +/- 0.0215 |
| `d18_step2_canonical` | 0.2379 +/- 0.0865 | 0.1508 +/- 0.0272 |  |  | 160.0000 +/- 0.0000 | 3.4028 +/- 0.1857 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0338 +/- 0.0092; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.5821 +/- 0.1049 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.4991 +/- 0.0310 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.5677 +/- 0.0434 |
| `d18_step2_canonical` | 0.8291 +/- 0.2045 | 0.9287 +/- 0.1832 |  |  | 160.0000 +/- 0.0000 | 3.0303 +/- 0.1736 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3193 +/- 0.0691; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.1541 +/- 0.4810 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.1160 +/- 0.4654 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.7826 +/- 0.0642 |
| `d18_step2_canonical` | 0.8417 +/- 0.5607 | 0.8467 +/- 0.2131 |  |  | 160.0000 +/- 0.0000 | 2.6263 +/- 0.0896 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1057 +/- 0.0838; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

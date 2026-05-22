# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 0.2072 +/- 0.0576 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 0.2470 +/- 0.0434 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 0.3366 +/- 0.0381 |
| `d18_step2_canonical` | 0.0712 +/- 0.0041 | 0.1305 +/- 0.0031 |  |  | 80.0000 +/- 0.0000 | 2.4516 +/- 0.0389 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0857 +/- 0.0282; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.3865 +/- 0.0647 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.2510 +/- 0.0418 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 0.4014 +/- 0.0286 |
| `d18_step2_canonical` | 0.0719 +/- 0.0106 | 0.2018 +/- 0.0225 |  |  | 80.0000 +/- 0.0000 | 2.6456 +/- 0.2308 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3563 +/- 0.0138; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0580 +/- 0.0034 | 0.1073 +/- 0.0013 |  |  |  | 0.8302 +/- 0.5710 |
| `mlp_h128` | 0.0751 +/- 0.0086 | 0.1187 +/- 0.0048 |  |  |  | 0.6967 +/- 0.3581 |
| `mlp_h64_64` | 0.0750 +/- 0.0060 | 0.1274 +/- 0.0034 |  |  |  | 0.5524 +/- 0.1894 |
| `d18_step2_canonical` | 0.0188 +/- 0.0019 | 0.0521 +/- 0.0030 |  |  | 80.0000 +/- 0.0000 | 2.6949 +/- 0.1213 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0392 +/- 0.0027; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8611 +/- 0.1893 | 0.9917 +/- 0.0835 |  |  |  | 0.2872 +/- 0.0503 |
| `mlp_h128` | 0.9468 +/- 0.2304 | 1.0382 +/- 0.0979 |  |  |  | 0.2153 +/- 0.0288 |
| `mlp_h64_64` | 0.9490 +/- 0.2284 | 1.0421 +/- 0.1050 |  |  |  | 0.3855 +/- 0.0492 |
| `d18_step2_canonical` | 0.3388 +/- 0.1074 | 0.4629 +/- 0.0640 |  |  | 80.0000 +/- 0.0000 | 2.2332 +/- 0.2871 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5211 +/- 0.0835; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.2846 +/- 0.0186 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.3173 +/- 0.0395 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.4806 +/- 0.0807 |
| `d18_step2_canonical` | 0.0207 +/- 0.0067 | 0.0553 +/- 0.0134 |  |  | 80.0000 +/- 0.0000 | 3.2292 +/- 0.2414 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0525 +/- 0.0052; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8637 +/- 0.0489 | 1.0904 +/- 0.0786 |  |  |  | 0.2417 +/- 0.0378 |
| `mlp_h128` | 0.8693 +/- 0.0485 | 1.0951 +/- 0.0761 |  |  |  | 0.2558 +/- 0.0420 |
| `mlp_h64_64` | 0.6059 +/- 0.0368 | 0.9161 +/- 0.0726 |  |  |  | 0.3333 +/- 0.0419 |
| `d18_step2_canonical` | 0.2768 +/- 0.0220 | 0.5168 +/- 0.0687 |  |  | 80.0000 +/- 0.0000 | 2.5181 +/- 0.0475 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3291 +/- 0.0313; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 0.5843 +/- 0.0265 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 0.6021 +/- 0.0575 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 0.6467 +/- 0.0261 |
| `d18_step2_canonical` | 0.0018 +/- 0.0002 | 0.0021 +/- 0.0001 | 0.9911 +/- 0.0011 | 0.8627 +/- 0.0160 | 80.0000 +/- 0.0000 | 4.0667 +/- 0.4909 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0012 +/- 0.0003; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.6970 +/- 0.0311; wins/losses/ties 3/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0310 +/- 0.0011 | 0.0440 +/- 0.0001 | 0.9133 +/- 0.0084 | 0.9264 +/- 0.0070 |  | 0.5840 +/- 0.0380 |
| `mlp_h128` | 0.0314 +/- 0.0010 | 0.0438 +/- 0.0003 | 0.9089 +/- 0.0056 | 0.9338 +/- 0.0016 |  | 0.7502 +/- 0.1543 |
| `mlp_h64_64` | 0.0332 +/- 0.0017 | 0.0478 +/- 0.0012 | 0.8789 +/- 0.0146 | 0.9109 +/- 0.0057 |  | 0.7143 +/- 0.0233 |
| `d18_step2_canonical` | 0.0270 +/- 0.0009 | 0.0344 +/- 0.0002 | 0.9311 +/- 0.0073 | 0.9450 +/- 0.0034 | 80.0000 +/- 0.0000 | 3.3588 +/- 0.1428 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0039 +/- 0.0003; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0105 +/- 0.0016; wins/losses/ties 3/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.0758 +/- 0.1386 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 0.8058 +/- 0.0298 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 1.0939 +/- 0.0831 |
| `d18_step2_canonical` | 0.0386 +/- 0.0007 | 0.0491 +/- 0.0007 | 0.8678 +/- 0.0149 | 0.9140 +/- 0.0091 | 80.0000 +/- 0.0000 | 6.0285 +/- 0.4895 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0005 +/- 0.0004; wins/losses/ties 1/2/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0068 +/- 0.0087; wins/losses/ties 2/1/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.7490 +/- 0.0196 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 1.0526 +/- 0.2988 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.9836 +/- 0.0499 |
| `d18_step2_canonical` | 0.0497 +/- 0.0019 | 0.0561 +/- 0.0012 | 0.7444 +/- 0.0223 | 0.7941 +/- 0.0129 | 80.0000 +/- 0.0000 | 5.7796 +/- 0.4888 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0019 +/- 0.0014; wins/losses/ties 1/2/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0260 +/- 0.0075; wins/losses/ties 0/3/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 0.8686 +/- 0.1390 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 0.6586 +/- 0.0215 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 0.6488 +/- 0.0554 |
| `d18_step2_canonical` | 0.0456 +/- 0.0014 | 0.0507 +/- 0.0002 | 0.8333 +/- 0.0203 | 0.8850 +/- 0.0070 | 80.0000 +/- 0.0000 | 4.4139 +/- 0.8182 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0035 +/- 0.0018; wins/losses/ties 2/1/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0037 +/- 0.0067; wins/losses/ties 2/1/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.7368 +/- 0.2302 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.7124 +/- 0.2791 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.3606 +/- 0.0318 |
| `d18_step2_canonical` | 0.2410 +/- 0.0920 | 0.1517 +/- 0.0290 |  |  | 80.0000 +/- 0.0000 | 2.4632 +/- 0.1381 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0307 +/- 0.0059; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.2624 +/- 0.0250 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.3046 +/- 0.0491 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.4098 +/- 0.0174 |
| `d18_step2_canonical` | 0.7992 +/- 0.1826 | 0.9064 +/- 0.1691 |  |  | 80.0000 +/- 0.0000 | 2.9787 +/- 0.1049 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3492 +/- 0.0790; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.7903 +/- 0.4807 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.5916 +/- 0.2678 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.4268 +/- 0.0490 |
| `d18_step2_canonical` | 0.8089 +/- 0.5393 | 0.8319 +/- 0.2103 |  |  | 80.0000 +/- 0.0000 | 2.6074 +/- 0.0937 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1386 +/- 0.1051; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

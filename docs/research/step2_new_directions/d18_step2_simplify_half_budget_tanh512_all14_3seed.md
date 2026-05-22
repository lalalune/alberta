# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 0.4675 +/- 0.0268 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 0.3524 +/- 0.0236 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 0.6163 +/- 0.0059 |
| `d18_step2_canonical` | 0.0706 +/- 0.0031 | 0.1367 +/- 0.0031 |  |  | 160.0000 +/- 0.0000 | 2.4914 +/- 0.0981 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0863 +/- 0.0237; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 0.4284 +/- 0.0412 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 0.3562 +/- 0.0175 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 0.3933 +/- 0.0245 |
| `d18_step2_canonical` | 0.0368 +/- 0.0042 | 0.1314 +/- 0.0165 |  |  | 160.0000 +/- 0.0000 | 3.9095 +/- 0.2849 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3915 +/- 0.0202; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0580 +/- 0.0034 | 0.1073 +/- 0.0013 |  |  |  | 0.9967 +/- 0.6471 |
| `mlp_h128` | 0.0751 +/- 0.0086 | 0.1187 +/- 0.0048 |  |  |  | 0.7771 +/- 0.5100 |
| `mlp_h64_64` | 0.0750 +/- 0.0060 | 0.1274 +/- 0.0034 |  |  |  | 0.8049 +/- 0.1647 |
| `d18_step2_canonical` | 0.0161 +/- 0.0008 | 0.0458 +/- 0.0011 |  |  | 160.0000 +/- 0.0000 | 4.8304 +/- 0.3224 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0418 +/- 0.0028; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8611 +/- 0.1893 | 0.9917 +/- 0.0835 |  |  |  | 0.2627 +/- 0.0323 |
| `mlp_h128` | 0.9468 +/- 0.2304 | 1.0382 +/- 0.0979 |  |  |  | 0.3932 +/- 0.0601 |
| `mlp_h64_64` | 0.9490 +/- 0.2284 | 1.0421 +/- 0.1050 |  |  |  | 0.7308 +/- 0.0495 |
| `d18_step2_canonical` | 0.1164 +/- 0.0393 | 0.3132 +/- 0.0342 |  |  | 160.0000 +/- 0.0000 | 2.5005 +/- 0.1413 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.7435 +/- 0.1527; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.2858 +/- 0.0332 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.3480 +/- 0.0431 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.5787 +/- 0.1150 |
| `d18_step2_canonical` | 0.0184 +/- 0.0052 | 0.0539 +/- 0.0133 |  |  | 160.0000 +/- 0.0000 | 2.2918 +/- 0.0621 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0547 +/- 0.0065; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8637 +/- 0.0489 | 1.0904 +/- 0.0786 |  |  |  | 0.2989 +/- 0.0389 |
| `mlp_h128` | 0.8693 +/- 0.0485 | 1.0951 +/- 0.0761 |  |  |  | 0.2856 +/- 0.0252 |
| `mlp_h64_64` | 0.6059 +/- 0.0368 | 0.9161 +/- 0.0726 |  |  |  | 0.6457 +/- 0.1202 |
| `d18_step2_canonical` | 0.0934 +/- 0.0078 | 0.3355 +/- 0.0345 |  |  | 160.0000 +/- 0.0000 | 2.0635 +/- 0.0274 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5125 +/- 0.0435; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 0.8255 +/- 0.0884 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 0.9451 +/- 0.1312 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 0.9070 +/- 0.0760 |
| `d18_step2_canonical` | 0.0016 +/- 0.0002 | 0.0019 +/- 0.0002 | 0.9922 +/- 0.0011 | 0.8627 +/- 0.0160 | 160.0000 +/- 0.0000 | 4.7751 +/- 0.3846 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0014 +/- 0.0003; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.6970 +/- 0.0311; wins/losses/ties 3/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0310 +/- 0.0011 | 0.0440 +/- 0.0001 | 0.9133 +/- 0.0084 | 0.9264 +/- 0.0070 |  | 1.2098 +/- 0.2836 |
| `mlp_h128` | 0.0314 +/- 0.0010 | 0.0438 +/- 0.0003 | 0.9089 +/- 0.0056 | 0.9338 +/- 0.0016 |  | 1.6037 +/- 0.7650 |
| `mlp_h64_64` | 0.0332 +/- 0.0017 | 0.0478 +/- 0.0012 | 0.8789 +/- 0.0146 | 0.9109 +/- 0.0057 |  | 1.1642 +/- 0.0952 |
| `d18_step2_canonical` | 0.0231 +/- 0.0008 | 0.0318 +/- 0.0003 | 0.9433 +/- 0.0019 | 0.9598 +/- 0.0016 | 160.0000 +/- 0.0000 | 4.6020 +/- 0.0163 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0078 +/- 0.0004; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0254 +/- 0.0012; wins/losses/ties 3/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 0.6132 +/- 0.0310 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 0.6844 +/- 0.0641 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 0.7182 +/- 0.0696 |
| `d18_step2_canonical` | 0.0352 +/- 0.0006 | 0.0470 +/- 0.0006 | 0.8756 +/- 0.0099 | 0.9412 +/- 0.0065 | 160.0000 +/- 0.0000 | 3.2350 +/- 0.3501 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0029 +/- 0.0004; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0340 +/- 0.0071; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.7163 +/- 0.0347 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 0.9531 +/- 0.0710 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.9729 +/- 0.0579 |
| `d18_step2_canonical` | 0.0465 +/- 0.0017 | 0.0546 +/- 0.0012 | 0.7811 +/- 0.0200 | 0.8077 +/- 0.0157 | 160.0000 +/- 0.0000 | 4.1320 +/- 0.1205 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0013 +/- 0.0010; wins/losses/ties 2/1/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0124 +/- 0.0055; wins/losses/ties 0/3/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 1.0449 +/- 0.0828 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 0.7915 +/- 0.0942 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 0.9955 +/- 0.0458 |
| `d18_step2_canonical` | 0.0418 +/- 0.0013 | 0.0481 +/- 0.0001 | 0.8544 +/- 0.0193 | 0.9004 +/- 0.0096 | 160.0000 +/- 0.0000 | 4.2286 +/- 0.2140 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0073 +/- 0.0013; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0192 +/- 0.0041; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.5884 +/- 0.7182 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.2751 +/- 0.3118 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.0115 +/- 0.0567 |
| `d18_step2_canonical` | 0.2326 +/- 0.0833 | 0.1485 +/- 0.0269 |  |  | 160.0000 +/- 0.0000 | 3.8057 +/- 0.1444 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0391 +/- 0.0128; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.5244 +/- 0.1194 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.5706 +/- 0.0238 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.7843 +/- 0.1144 |
| `d18_step2_canonical` | 0.8233 +/- 0.2087 | 0.9250 +/- 0.1795 |  |  | 160.0000 +/- 0.0000 | 3.1343 +/- 0.1472 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.3251 +/- 0.0647; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.3678 +/- 0.6033 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.0872 +/- 0.5342 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.8125 +/- 0.1290 |
| `d18_step2_canonical` | 0.8453 +/- 0.5653 | 0.8500 +/- 0.2139 |  |  | 160.0000 +/- 0.0000 | 2.7710 +/- 0.0629 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1022 +/- 0.0791; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

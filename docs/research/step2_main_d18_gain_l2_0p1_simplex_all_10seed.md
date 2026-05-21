# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_gain_l2_0p1.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3952 +/- 0.0203 | 0.5573 +/- 0.0111 |  |  |  | 0.1699 +/- 0.0127 |
| `mlp_h128` | 0.4233 +/- 0.0144 | 0.5731 +/- 0.0089 |  |  |  | 0.2086 +/- 0.0328 |
| `mlp_h64_64` | 0.1811 +/- 0.0157 | 0.3884 +/- 0.0142 |  |  |  | 0.2277 +/- 0.0141 |
| `d18_step2_gain_l2_0p1` | 0.0353 +/- 0.0005 | 0.0900 +/- 0.0007 |  |  | 320.0000 +/- 0.0000 | 1.8777 +/- 0.1584 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1458 +/- 0.0157; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4929 +/- 0.0342 | 0.6412 +/- 0.0189 |  |  |  | 0.1845 +/- 0.0164 |
| `mlp_h128` | 0.5657 +/- 0.0339 | 0.6617 +/- 0.0220 |  |  |  | 0.1720 +/- 0.0081 |
| `mlp_h64_64` | 0.6451 +/- 0.0394 | 0.7386 +/- 0.0285 |  |  |  | 0.2211 +/- 0.0148 |
| `d18_step2_gain_l2_0p1` | 0.0423 +/- 0.0040 | 0.1153 +/- 0.0103 |  |  | 320.0000 +/- 0.0000 | 1.7726 +/- 0.0780 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4506 +/- 0.0307; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0707 +/- 0.0041 | 0.1097 +/- 0.0032 |  |  |  | 0.2755 +/- 0.1118 |
| `mlp_h128` | 0.0872 +/- 0.0053 | 0.1191 +/- 0.0029 |  |  |  | 0.2730 +/- 0.0947 |
| `mlp_h64_64` | 0.0952 +/- 0.0076 | 0.1309 +/- 0.0031 |  |  |  | 0.3177 +/- 0.0611 |
| `d18_step2_gain_l2_0p1` | 0.0196 +/- 0.0011 | 0.0452 +/- 0.0010 |  |  | 320.0000 +/- 0.0000 | 1.8851 +/- 0.1416 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0511 +/- 0.0037; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9111 +/- 0.0868 | 1.0747 +/- 0.0434 |  |  |  | 0.1710 +/- 0.0252 |
| `mlp_h128` | 1.0331 +/- 0.0969 | 1.1401 +/- 0.0493 |  |  |  | 0.1587 +/- 0.0139 |
| `mlp_h64_64` | 1.0520 +/- 0.0993 | 1.1557 +/- 0.0536 |  |  |  | 0.1919 +/- 0.0156 |
| `d18_step2_gain_l2_0p1` | 0.0950 +/- 0.0172 | 0.3234 +/- 0.0279 |  |  | 320.0000 +/- 0.0000 | 1.6310 +/- 0.1154 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.8157 +/- 0.0754; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0938 +/- 0.0115 | 0.1103 +/- 0.0080 |  |  |  | 0.1649 +/- 0.0075 |
| `mlp_h128` | 0.0981 +/- 0.0112 | 0.1134 +/- 0.0076 |  |  |  | 0.1864 +/- 0.0125 |
| `mlp_h64_64` | 0.1150 +/- 0.0110 | 0.1292 +/- 0.0077 |  |  |  | 0.2344 +/- 0.0188 |
| `d18_step2_gain_l2_0p1` | 0.0350 +/- 0.0074 | 0.0524 +/- 0.0054 |  |  | 320.0000 +/- 0.0000 | 1.4491 +/- 0.0396 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0586 +/- 0.0053; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0086 +/- 0.0515 | 1.1263 +/- 0.0416 |  |  |  | 0.2185 +/- 0.0118 |
| `mlp_h128` | 1.0193 +/- 0.0526 | 1.1366 +/- 0.0417 |  |  |  | 0.1685 +/- 0.0063 |
| `mlp_h64_64` | 0.7461 +/- 0.0462 | 0.9548 +/- 0.0445 |  |  |  | 0.2289 +/- 0.0138 |
| `d18_step2_gain_l2_0p1` | 0.0547 +/- 0.0048 | 0.2790 +/- 0.0243 |  |  | 320.0000 +/- 0.0000 | 1.7916 +/- 0.0530 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.6914 +/- 0.0422; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.5418 +/- 0.0492 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.5671 +/- 0.0381 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.6338 +/- 0.0575 |
| `d18_step2_gain_l2_0p1` | 0.0029 +/- 0.0003 | 0.0031 +/- 0.0001 | 0.9857 +/- 0.0013 | 0.1987 +/- 0.0140 | 320.0000 +/- 0.0000 | 3.5238 +/- 0.5259 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0000 +/- 0.0003; wins/losses/ties 5/5/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0629 +/- 0.0103; wins/losses/ties 10/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0315 +/- 0.0005 | 0.0452 +/- 0.0004 | 0.9123 +/- 0.0053 | 0.9148 +/- 0.0041 |  | 0.5312 +/- 0.0531 |
| `mlp_h128` | 0.0306 +/- 0.0007 | 0.0438 +/- 0.0003 | 0.9257 +/- 0.0056 | 0.9306 +/- 0.0045 |  | 0.6311 +/- 0.0961 |
| `mlp_h64_64` | 0.0325 +/- 0.0006 | 0.0486 +/- 0.0004 | 0.8900 +/- 0.0058 | 0.9058 +/- 0.0061 |  | 0.6968 +/- 0.1338 |
| `d18_step2_gain_l2_0p1` | 0.0078 +/- 0.0006 | 0.0197 +/- 0.0004 | 0.9610 +/- 0.0031 | 0.9633 +/- 0.0037 | 320.0000 +/- 0.0000 | 3.6426 +/- 0.6287 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0226 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0323 +/- 0.0034; wins/losses/ties 10/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 0.3736 +/- 0.0147 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 0.3930 +/- 0.0125 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 0.4278 +/- 0.0151 |
| `d18_step2_gain_l2_0p1` | 0.0224 +/- 0.0011 | 0.0439 +/- 0.0009 | 0.8880 +/- 0.0057 | 0.9475 +/- 0.0038 | 320.0000 +/- 0.0000 | 2.0674 +/- 0.0575 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0172 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0475 +/- 0.0055; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.3517 +/- 0.0160 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.3692 +/- 0.0106 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.4522 +/- 0.0364 |
| `d18_step2_gain_l2_0p1` | 0.0389 +/- 0.0028 | 0.0550 +/- 0.0020 | 0.8057 +/- 0.0142 | 0.8458 +/- 0.0119 | 320.0000 +/- 0.0000 | 1.9842 +/- 0.0425 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0091 +/- 0.0020; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0098 +/- 0.0094; wins/losses/ties 6/4/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 0.5062 +/- 0.0474 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 0.4691 +/- 0.0205 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 0.5344 +/- 0.0246 |
| `d18_step2_gain_l2_0p1` | 0.0241 +/- 0.0007 | 0.0402 +/- 0.0003 | 0.8793 +/- 0.0037 | 0.9217 +/- 0.0047 | 320.0000 +/- 0.0000 | 2.6412 +/- 0.0794 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0242 +/- 0.0006; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0436 +/- 0.0048; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.2757 +/- 0.0509 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.2589 +/- 0.0412 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.2673 +/- 0.0134 |
| `d18_step2_gain_l2_0p1` | 0.2271 +/- 0.0284 | 0.1784 +/- 0.0141 |  |  | 320.0000 +/- 0.0000 | 1.7414 +/- 0.0512 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0422 +/- 0.0179; wins/losses/ties 9/1/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.6271 +/- 0.3675 | 1.4277 +/- 0.1462 |  |  |  | 0.1546 +/- 0.0066 |
| `mlp_h128` | 1.6349 +/- 0.3642 | 1.4399 +/- 0.1466 |  |  |  | 0.1483 +/- 0.0069 |
| `mlp_h64_64` | 1.6000 +/- 0.3571 | 1.4221 +/- 0.1407 |  |  |  | 0.2135 +/- 0.0153 |
| `d18_step2_gain_l2_0p1` | 1.0670 +/- 0.2466 | 0.8905 +/- 0.0955 |  |  | 320.0000 +/- 0.0000 | 1.3619 +/- 0.0291 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5270 +/- 0.1185; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.2157 | 1.0156 +/- 0.0967 |  |  |  | 0.4146 +/- 0.0785 |
| `mlp_h128` | 1.0368 +/- 0.2152 | 1.0187 +/- 0.0969 |  |  |  | 0.3955 +/- 0.0688 |
| `mlp_h64_64` | 1.0051 +/- 0.2125 | 0.9966 +/- 0.0942 |  |  |  | 0.4698 +/- 0.0642 |
| `d18_step2_gain_l2_0p1` | 0.9110 +/- 0.1903 | 0.8196 +/- 0.0856 |  |  | 320.0000 +/- 0.0000 | 4.1348 +/- 0.7703 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0941 +/- 0.0306; wins/losses/ties 9/1/0; best-D18 counts {'d18_step2_gain_l2_0p1': 10}.

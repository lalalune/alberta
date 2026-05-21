# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3952 +/- 0.0203 | 0.5573 +/- 0.0111 |  |  |  | 0.1080 +/- 0.0014 |
| `mlp_h128` | 0.4233 +/- 0.0144 | 0.5731 +/- 0.0089 |  |  |  | 0.1122 +/- 0.0059 |
| `mlp_h64_64` | 0.1811 +/- 0.0157 | 0.3884 +/- 0.0142 |  |  |  | 0.1436 +/- 0.0019 |
| `d18_step2_canonical` | 0.0376 +/- 0.0009 | 0.1105 +/- 0.0010 |  |  | 320.0000 +/- 0.0000 | 1.1527 +/- 0.0040 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1435 +/- 0.0158; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4929 +/- 0.0342 | 0.6412 +/- 0.0189 |  |  |  | 0.1239 +/- 0.0077 |
| `mlp_h128` | 0.5657 +/- 0.0339 | 0.6617 +/- 0.0220 |  |  |  | 0.1199 +/- 0.0052 |
| `mlp_h64_64` | 0.6451 +/- 0.0394 | 0.7386 +/- 0.0285 |  |  |  | 0.1574 +/- 0.0033 |
| `d18_step2_canonical` | 0.0397 +/- 0.0042 | 0.1078 +/- 0.0104 |  |  | 320.0000 +/- 0.0000 | 1.2067 +/- 0.0071 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4532 +/- 0.0307; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0707 +/- 0.0041 | 0.1097 +/- 0.0032 |  |  |  | 0.1913 +/- 0.0737 |
| `mlp_h128` | 0.0872 +/- 0.0053 | 0.1191 +/- 0.0029 |  |  |  | 0.1841 +/- 0.0737 |
| `mlp_h64_64` | 0.0952 +/- 0.0076 | 0.1309 +/- 0.0031 |  |  |  | 0.1919 +/- 0.0346 |
| `d18_step2_canonical` | 0.0194 +/- 0.0010 | 0.0460 +/- 0.0009 |  |  | 320.0000 +/- 0.0000 | 1.2079 +/- 0.0171 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0513 +/- 0.0038; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9111 +/- 0.0868 | 1.0747 +/- 0.0434 |  |  |  | 0.1131 +/- 0.0077 |
| `mlp_h128` | 1.0331 +/- 0.0969 | 1.1401 +/- 0.0493 |  |  |  | 0.1249 +/- 0.0158 |
| `mlp_h64_64` | 1.0520 +/- 0.0993 | 1.1557 +/- 0.0536 |  |  |  | 0.1687 +/- 0.0246 |
| `d18_step2_canonical` | 0.0686 +/- 0.0110 | 0.3177 +/- 0.0297 |  |  | 320.0000 +/- 0.0000 | 1.2212 +/- 0.0437 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.8421 +/- 0.0782; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0938 +/- 0.0115 | 0.1103 +/- 0.0080 |  |  |  | 0.1295 +/- 0.0046 |
| `mlp_h128` | 0.0981 +/- 0.0112 | 0.1134 +/- 0.0076 |  |  |  | 0.1253 +/- 0.0021 |
| `mlp_h64_64` | 0.1150 +/- 0.0110 | 0.1292 +/- 0.0077 |  |  |  | 0.1826 +/- 0.0111 |
| `d18_step2_canonical` | 0.0329 +/- 0.0073 | 0.0514 +/- 0.0053 |  |  | 320.0000 +/- 0.0000 | 1.1740 +/- 0.0035 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0608 +/- 0.0052; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0086 +/- 0.0515 | 1.1263 +/- 0.0416 |  |  |  | 0.1085 +/- 0.0049 |
| `mlp_h128` | 1.0193 +/- 0.0526 | 1.1366 +/- 0.0417 |  |  |  | 0.1108 +/- 0.0081 |
| `mlp_h64_64` | 0.7461 +/- 0.0462 | 0.9548 +/- 0.0445 |  |  |  | 0.1455 +/- 0.0056 |
| `d18_step2_canonical` | 0.0557 +/- 0.0056 | 0.2679 +/- 0.0238 |  |  | 320.0000 +/- 0.0000 | 1.1712 +/- 0.0087 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.6904 +/- 0.0417; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.3071 +/- 0.0185 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.3080 +/- 0.0143 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.3698 +/- 0.0332 |
| `d18_step2_canonical` | 0.0016 +/- 0.0001 | 0.0020 +/- 0.0001 | 0.9920 +/- 0.0005 | 0.8505 +/- 0.0066 | 320.0000 +/- 0.0000 | 2.0600 +/- 0.1158 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0013 +/- 0.0001; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.7147 +/- 0.0134; wins/losses/ties 10/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0315 +/- 0.0005 | 0.0452 +/- 0.0004 | 0.9123 +/- 0.0053 | 0.9148 +/- 0.0041 |  | 0.2934 +/- 0.0152 |
| `mlp_h128` | 0.0306 +/- 0.0007 | 0.0438 +/- 0.0003 | 0.9257 +/- 0.0056 | 0.9306 +/- 0.0045 |  | 0.3136 +/- 0.0282 |
| `mlp_h64_64` | 0.0325 +/- 0.0006 | 0.0486 +/- 0.0004 | 0.8900 +/- 0.0058 | 0.9058 +/- 0.0061 |  | 0.3308 +/- 0.0167 |
| `d18_step2_canonical` | 0.0212 +/- 0.0003 | 0.0303 +/- 0.0002 | 0.9593 +/- 0.0033 | 0.9642 +/- 0.0034 | 320.0000 +/- 0.0000 | 2.1727 +/- 0.3487 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0092 +/- 0.0005; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0332 +/- 0.0034; wins/losses/ties 10/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 0.2765 +/- 0.0011 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 0.2797 +/- 0.0016 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 0.3062 +/- 0.0012 |
| `d18_step2_canonical` | 0.0342 +/- 0.0005 | 0.0465 +/- 0.0004 | 0.8857 +/- 0.0055 | 0.9455 +/- 0.0034 | 320.0000 +/- 0.0000 | 1.9493 +/- 0.0109 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0054 +/- 0.0005; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0455 +/- 0.0053; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.2781 +/- 0.0016 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.2810 +/- 0.0007 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.3358 +/- 0.0265 |
| `d18_step2_canonical` | 0.0434 +/- 0.0015 | 0.0528 +/- 0.0009 | 0.8073 +/- 0.0151 | 0.8469 +/- 0.0125 | 320.0000 +/- 0.0000 | 1.9774 +/- 0.0075 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0046 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0109 +/- 0.0093; wins/losses/ties 6/4/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 0.3107 +/- 0.0237 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 0.2902 +/- 0.0022 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 0.3210 +/- 0.0028 |
| `d18_step2_canonical` | 0.0384 +/- 0.0005 | 0.0464 +/- 0.0002 | 0.8783 +/- 0.0047 | 0.9223 +/- 0.0056 | 320.0000 +/- 0.0000 | 2.1060 +/- 0.1755 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0100 +/- 0.0004; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0442 +/- 0.0054; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.1859 +/- 0.0357 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.1832 +/- 0.0340 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.1922 +/- 0.0022 |
| `d18_step2_canonical` | 0.2110 +/- 0.0285 | 0.1711 +/- 0.0133 |  |  | 320.0000 +/- 0.0000 | 1.3034 +/- 0.0251 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0584 +/- 0.0208; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.6271 +/- 0.3675 | 1.4277 +/- 0.1462 |  |  |  | 0.1380 +/- 0.0047 |
| `mlp_h128` | 1.6349 +/- 0.3642 | 1.4399 +/- 0.1466 |  |  |  | 0.1376 +/- 0.0040 |
| `mlp_h64_64` | 1.6000 +/- 0.3571 | 1.4221 +/- 0.1407 |  |  |  | 0.1910 +/- 0.0147 |
| `d18_step2_canonical` | 1.0736 +/- 0.2699 | 0.9549 +/- 0.1026 |  |  | 320.0000 +/- 0.0000 | 1.2149 +/- 0.0167 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5204 +/- 0.0948; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.2157 | 1.0156 +/- 0.0967 |  |  |  | 0.2023 +/- 0.0515 |
| `mlp_h128` | 1.0368 +/- 0.2152 | 1.0187 +/- 0.0969 |  |  |  | 0.1945 +/- 0.0491 |
| `mlp_h64_64` | 1.0051 +/- 0.2125 | 0.9966 +/- 0.0942 |  |  |  | 0.1883 +/- 0.0029 |
| `d18_step2_canonical` | 0.8848 +/- 0.1899 | 0.7969 +/- 0.0839 |  |  | 320.0000 +/- 0.0000 | 1.4306 +/- 0.0129 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1203 +/- 0.0307; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

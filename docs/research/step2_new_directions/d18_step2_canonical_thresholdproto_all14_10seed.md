# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3952 +/- 0.0203 | 0.5573 +/- 0.0111 |  |  |  | 0.1324 +/- 0.0076 |
| `mlp_h128` | 0.4233 +/- 0.0144 | 0.5731 +/- 0.0089 |  |  |  | 0.1190 +/- 0.0027 |
| `mlp_h64_64` | 0.1811 +/- 0.0157 | 0.3884 +/- 0.0142 |  |  |  | 0.1875 +/- 0.0079 |
| `d18_step2_canonical` | 0.0438 +/- 0.0014 | 0.1268 +/- 0.0012 |  |  | 320.0000 +/- 0.0000 | 1.3430 +/- 0.0115 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1373 +/- 0.0156; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4929 +/- 0.0342 | 0.6412 +/- 0.0189 |  |  |  | 0.1255 +/- 0.0073 |
| `mlp_h128` | 0.5657 +/- 0.0339 | 0.6617 +/- 0.0220 |  |  |  | 0.1144 +/- 0.0092 |
| `mlp_h64_64` | 0.6451 +/- 0.0394 | 0.7386 +/- 0.0285 |  |  |  | 0.1634 +/- 0.0064 |
| `d18_step2_canonical` | 0.0395 +/- 0.0042 | 0.1081 +/- 0.0106 |  |  | 320.0000 +/- 0.0000 | 1.3070 +/- 0.0513 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4534 +/- 0.0306; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0707 +/- 0.0041 | 0.1097 +/- 0.0032 |  |  |  | 0.1753 +/- 0.0702 |
| `mlp_h128` | 0.0872 +/- 0.0053 | 0.1191 +/- 0.0029 |  |  |  | 0.1871 +/- 0.0618 |
| `mlp_h64_64` | 0.0952 +/- 0.0076 | 0.1309 +/- 0.0031 |  |  |  | 0.1775 +/- 0.0321 |
| `d18_step2_canonical` | 0.0197 +/- 0.0011 | 0.0460 +/- 0.0009 |  |  | 320.0000 +/- 0.0000 | 1.2389 +/- 0.0154 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0510 +/- 0.0038; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9111 +/- 0.0868 | 1.0747 +/- 0.0434 |  |  |  | 0.1346 +/- 0.0077 |
| `mlp_h128` | 1.0331 +/- 0.0969 | 1.1401 +/- 0.0493 |  |  |  | 0.1283 +/- 0.0066 |
| `mlp_h64_64` | 1.0520 +/- 0.0993 | 1.1557 +/- 0.0536 |  |  |  | 0.1828 +/- 0.0086 |
| `d18_step2_canonical` | 0.0679 +/- 0.0111 | 0.3174 +/- 0.0295 |  |  | 320.0000 +/- 0.0000 | 1.3798 +/- 0.0262 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.8428 +/- 0.0785; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0938 +/- 0.0115 | 0.1103 +/- 0.0080 |  |  |  | 0.1397 +/- 0.0200 |
| `mlp_h128` | 0.0981 +/- 0.0112 | 0.1134 +/- 0.0076 |  |  |  | 0.1341 +/- 0.0084 |
| `mlp_h64_64` | 0.1150 +/- 0.0110 | 0.1292 +/- 0.0077 |  |  |  | 0.1779 +/- 0.0103 |
| `d18_step2_canonical` | 0.0335 +/- 0.0076 | 0.0518 +/- 0.0054 |  |  | 320.0000 +/- 0.0000 | 1.4040 +/- 0.1321 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0602 +/- 0.0050; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0086 +/- 0.0515 | 1.1263 +/- 0.0416 |  |  |  | 0.1332 +/- 0.0121 |
| `mlp_h128` | 1.0193 +/- 0.0526 | 1.1366 +/- 0.0417 |  |  |  | 0.1029 +/- 0.0021 |
| `mlp_h64_64` | 0.7461 +/- 0.0462 | 0.9548 +/- 0.0445 |  |  |  | 0.1750 +/- 0.0132 |
| `d18_step2_canonical` | 0.0537 +/- 0.0053 | 0.2678 +/- 0.0238 |  |  | 320.0000 +/- 0.0000 | 1.2420 +/- 0.0143 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.6924 +/- 0.0420; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.3721 +/- 0.0141 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.4038 +/- 0.0114 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.4748 +/- 0.0271 |
| `d18_step2_canonical` | 0.0016 +/- 0.0001 | 0.0021 +/- 0.0001 | 0.9920 +/- 0.0007 | 0.8510 +/- 0.0060 | 320.0000 +/- 0.0000 | 2.5268 +/- 0.1246 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0013 +/- 0.0002; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.7152 +/- 0.0131; wins/losses/ties 10/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0315 +/- 0.0005 | 0.0452 +/- 0.0004 | 0.9123 +/- 0.0053 | 0.9148 +/- 0.0041 |  | 0.4046 +/- 0.0284 |
| `mlp_h128` | 0.0306 +/- 0.0007 | 0.0438 +/- 0.0003 | 0.9257 +/- 0.0056 | 0.9306 +/- 0.0045 |  | 0.4420 +/- 0.0420 |
| `mlp_h64_64` | 0.0325 +/- 0.0006 | 0.0486 +/- 0.0004 | 0.8900 +/- 0.0058 | 0.9058 +/- 0.0061 |  | 0.4646 +/- 0.0296 |
| `d18_step2_canonical` | 0.0209 +/- 0.0003 | 0.0300 +/- 0.0002 | 0.9607 +/- 0.0033 | 0.9631 +/- 0.0039 | 320.0000 +/- 0.0000 | 2.4347 +/- 0.0765 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0095 +/- 0.0005; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0321 +/- 0.0036; wins/losses/ties 10/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 0.3286 +/- 0.0141 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 0.3362 +/- 0.0087 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 0.3510 +/- 0.0085 |
| `d18_step2_canonical` | 0.0338 +/- 0.0005 | 0.0462 +/- 0.0004 | 0.8870 +/- 0.0056 | 0.9456 +/- 0.0043 | 320.0000 +/- 0.0000 | 2.0030 +/- 0.0242 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0058 +/- 0.0005; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0456 +/- 0.0058; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.3224 +/- 0.0126 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.3458 +/- 0.0119 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.4116 +/- 0.0401 |
| `d18_step2_canonical` | 0.0431 +/- 0.0015 | 0.0526 +/- 0.0009 | 0.8063 +/- 0.0135 | 0.8456 +/- 0.0122 | 320.0000 +/- 0.0000 | 2.0504 +/- 0.0343 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0049 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0096 +/- 0.0093; wins/losses/ties 6/4/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 0.3754 +/- 0.0357 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 0.3479 +/- 0.0171 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 0.3662 +/- 0.0155 |
| `d18_step2_canonical` | 0.0379 +/- 0.0005 | 0.0461 +/- 0.0001 | 0.8747 +/- 0.0043 | 0.9206 +/- 0.0048 | 320.0000 +/- 0.0000 | 2.0521 +/- 0.0591 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0104 +/- 0.0004; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0425 +/- 0.0051; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.2637 +/- 0.0425 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.2569 +/- 0.0422 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.2657 +/- 0.0098 |
| `d18_step2_canonical` | 0.2064 +/- 0.0277 | 0.1683 +/- 0.0129 |  |  | 320.0000 +/- 0.0000 | 1.7640 +/- 0.0281 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0629 +/- 0.0210; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.6271 +/- 0.3675 | 1.4277 +/- 0.1462 |  |  |  | 0.1604 +/- 0.0086 |
| `mlp_h128` | 1.6349 +/- 0.3642 | 1.4399 +/- 0.1466 |  |  |  | 0.1522 +/- 0.0046 |
| `mlp_h64_64` | 1.6000 +/- 0.3571 | 1.4221 +/- 0.1407 |  |  |  | 0.2259 +/- 0.0137 |
| `d18_step2_canonical` | 1.0696 +/- 0.2687 | 0.9515 +/- 0.1029 |  |  | 320.0000 +/- 0.0000 | 1.5372 +/- 0.1275 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5244 +/- 0.0975; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.2157 | 1.0156 +/- 0.0967 |  |  |  | 0.2295 +/- 0.0536 |
| `mlp_h128` | 1.0368 +/- 0.2152 | 1.0187 +/- 0.0969 |  |  |  | 0.2281 +/- 0.0618 |
| `mlp_h64_64` | 1.0051 +/- 0.2125 | 0.9966 +/- 0.0942 |  |  |  | 0.2265 +/- 0.0122 |
| `d18_step2_canonical` | 0.8850 +/- 0.1898 | 0.7987 +/- 0.0836 |  |  | 320.0000 +/- 0.0000 | 1.5286 +/- 0.0454 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1201 +/- 0.0303; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3952 +/- 0.0203 | 0.5573 +/- 0.0111 |  |  |  | 0.3566 +/- 0.0353 |
| `mlp_h128` | 0.4233 +/- 0.0144 | 0.5731 +/- 0.0089 |  |  |  | 0.3515 +/- 0.0405 |
| `mlp_h64_64` | 0.1811 +/- 0.0157 | 0.3884 +/- 0.0142 |  |  |  | 0.4589 +/- 0.0295 |
| `d18_step2_canonical` | 0.0491 +/- 0.0014 | 0.1139 +/- 0.0012 |  |  | 160.0000 +/- 0.0000 | 4.7147 +/- 0.3297 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1320 +/- 0.0159; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4929 +/- 0.0342 | 0.6412 +/- 0.0189 |  |  |  | 0.3679 +/- 0.0378 |
| `mlp_h128` | 0.5657 +/- 0.0339 | 0.6617 +/- 0.0220 |  |  |  | 0.4063 +/- 0.0286 |
| `mlp_h64_64` | 0.6451 +/- 0.0394 | 0.7386 +/- 0.0285 |  |  |  | 0.4888 +/- 0.0477 |
| `d18_step2_canonical` | 0.0487 +/- 0.0044 | 0.1298 +/- 0.0106 |  |  | 160.0000 +/- 0.0000 | 4.5636 +/- 0.3723 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4442 +/- 0.0306; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0707 +/- 0.0041 | 0.1097 +/- 0.0032 |  |  |  | 0.5903 +/- 0.2431 |
| `mlp_h128` | 0.0872 +/- 0.0053 | 0.1191 +/- 0.0029 |  |  |  | 0.6131 +/- 0.2004 |
| `mlp_h64_64` | 0.0952 +/- 0.0076 | 0.1309 +/- 0.0031 |  |  |  | 0.5527 +/- 0.1091 |
| `d18_step2_canonical` | 0.0187 +/- 0.0009 | 0.0477 +/- 0.0013 |  |  | 160.0000 +/- 0.0000 | 4.2517 +/- 0.4556 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0520 +/- 0.0037; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9111 +/- 0.0868 | 1.0747 +/- 0.0434 |  |  |  | 0.3455 +/- 0.0216 |
| `mlp_h128` | 1.0331 +/- 0.0969 | 1.1401 +/- 0.0493 |  |  |  | 0.3366 +/- 0.0296 |
| `mlp_h64_64` | 1.0520 +/- 0.0993 | 1.1557 +/- 0.0536 |  |  |  | 0.4892 +/- 0.0442 |
| `d18_step2_canonical` | 0.1452 +/- 0.0267 | 0.3916 +/- 0.0332 |  |  | 160.0000 +/- 0.0000 | 4.0491 +/- 0.2943 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.7656 +/- 0.0626; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0938 +/- 0.0115 | 0.1103 +/- 0.0080 |  |  |  | 0.5920 +/- 0.0717 |
| `mlp_h128` | 0.0981 +/- 0.0112 | 0.1134 +/- 0.0076 |  |  |  | 0.5162 +/- 0.0543 |
| `mlp_h64_64` | 0.1150 +/- 0.0110 | 0.1292 +/- 0.0077 |  |  |  | 0.8965 +/- 0.1027 |
| `d18_step2_canonical` | 0.0342 +/- 0.0080 | 0.0525 +/- 0.0055 |  |  | 160.0000 +/- 0.0000 | 5.8797 +/- 0.3885 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0595 +/- 0.0050; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0086 +/- 0.0515 | 1.1263 +/- 0.0416 |  |  |  | 0.5434 +/- 0.0720 |
| `mlp_h128` | 1.0193 +/- 0.0526 | 1.1366 +/- 0.0417 |  |  |  | 0.5644 +/- 0.0834 |
| `mlp_h64_64` | 0.7461 +/- 0.0462 | 0.9548 +/- 0.0445 |  |  |  | 0.7561 +/- 0.0766 |
| `d18_step2_canonical` | 0.1408 +/- 0.0270 | 0.3638 +/- 0.0280 |  |  | 160.0000 +/- 0.0000 | 7.4268 +/- 0.5276 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.6053 +/- 0.0326; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.5218 +/- 0.0392 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.5619 +/- 0.0464 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.6565 +/- 0.0614 |
| `d18_step2_canonical` | 0.0016 +/- 0.0001 | 0.0021 +/- 0.0001 | 0.9920 +/- 0.0005 | 0.8518 +/- 0.0067 | 160.0000 +/- 0.0000 | 3.3957 +/- 0.6419 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0013 +/- 0.0001; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.7160 +/- 0.0131; wins/losses/ties 10/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0315 +/- 0.0005 | 0.0452 +/- 0.0004 | 0.9123 +/- 0.0053 | 0.9148 +/- 0.0041 |  | 0.6281 +/- 0.0417 |
| `mlp_h128` | 0.0306 +/- 0.0007 | 0.0438 +/- 0.0003 | 0.9257 +/- 0.0056 | 0.9306 +/- 0.0045 |  | 0.6482 +/- 0.0449 |
| `mlp_h64_64` | 0.0325 +/- 0.0006 | 0.0486 +/- 0.0004 | 0.8900 +/- 0.0058 | 0.9058 +/- 0.0061 |  | 0.6886 +/- 0.0641 |
| `d18_step2_canonical` | 0.0234 +/- 0.0004 | 0.0323 +/- 0.0002 | 0.9480 +/- 0.0023 | 0.9536 +/- 0.0033 | 160.0000 +/- 0.0000 | 2.6730 +/- 0.1967 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0071 +/- 0.0005; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0226 +/- 0.0037; wins/losses/ties 10/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 0.4965 +/- 0.0308 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 0.4859 +/- 0.0338 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 0.5495 +/- 0.0422 |
| `d18_step2_canonical` | 0.0357 +/- 0.0006 | 0.0477 +/- 0.0004 | 0.8767 +/- 0.0041 | 0.9341 +/- 0.0035 | 160.0000 +/- 0.0000 | 2.4336 +/- 0.4155 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0039 +/- 0.0005; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0341 +/- 0.0049; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.6462 +/- 0.1016 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.6015 +/- 0.0762 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.7616 +/- 0.0976 |
| `d18_step2_canonical` | 0.0446 +/- 0.0015 | 0.0538 +/- 0.0008 | 0.7957 +/- 0.0146 | 0.8306 +/- 0.0132 | 160.0000 +/- 0.0000 | 3.3503 +/- 0.5283 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0034 +/- 0.0009; wins/losses/ties 9/1/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0054 +/- 0.0088; wins/losses/ties 6/4/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 0.6306 +/- 0.0672 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 0.6892 +/- 0.1130 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 0.7449 +/- 0.0919 |
| `d18_step2_canonical` | 0.0418 +/- 0.0005 | 0.0483 +/- 0.0002 | 0.8580 +/- 0.0062 | 0.9013 +/- 0.0051 | 160.0000 +/- 0.0000 | 4.0758 +/- 0.7359 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0065 +/- 0.0004; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0232 +/- 0.0056; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.3226 +/- 0.0691 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.3239 +/- 0.0744 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.3432 +/- 0.0182 |
| `d18_step2_canonical` | 0.2131 +/- 0.0305 | 0.1742 +/- 0.0138 |  |  | 160.0000 +/- 0.0000 | 1.9451 +/- 0.0929 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0562 +/- 0.0192; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.6271 +/- 0.3675 | 1.4277 +/- 0.1462 |  |  |  | 0.3381 +/- 0.0395 |
| `mlp_h128` | 1.6349 +/- 0.3642 | 1.4399 +/- 0.1466 |  |  |  | 0.3422 +/- 0.0287 |
| `mlp_h64_64` | 1.6000 +/- 0.3571 | 1.4221 +/- 0.1407 |  |  |  | 0.5008 +/- 0.0281 |
| `d18_step2_canonical` | 1.0721 +/- 0.2715 | 0.9386 +/- 0.1027 |  |  | 160.0000 +/- 0.0000 | 5.0249 +/- 0.3077 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5219 +/- 0.0955; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.2157 | 1.0156 +/- 0.0967 |  |  |  | 0.7282 +/- 0.2199 |
| `mlp_h128` | 1.0368 +/- 0.2152 | 1.0187 +/- 0.0969 |  |  |  | 0.6640 +/- 0.1958 |
| `mlp_h64_64` | 1.0051 +/- 0.2125 | 0.9966 +/- 0.0942 |  |  |  | 0.7123 +/- 0.0664 |
| `d18_step2_canonical` | 0.8888 +/- 0.1936 | 0.8008 +/- 0.0841 |  |  | 160.0000 +/- 0.0000 | 7.2113 +/- 0.5498 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1163 +/- 0.0259; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

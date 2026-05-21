# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3952 +/- 0.0203 | 0.5573 +/- 0.0111 |  |  |  | 0.8686 +/- 0.2693 |
| `mlp_h128` | 0.4233 +/- 0.0144 | 0.5731 +/- 0.0089 |  |  |  | 0.6928 +/- 0.1539 |
| `mlp_h64_64` | 0.1811 +/- 0.0157 | 0.3884 +/- 0.0142 |  |  |  | 0.8336 +/- 0.0855 |
| `d18_step2_canonical` | 0.0449 +/- 0.0016 | 0.1284 +/- 0.0013 |  |  | 320.0000 +/- 0.0000 | 3.9184 +/- 0.2281 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1362 +/- 0.0155; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4929 +/- 0.0342 | 0.6412 +/- 0.0189 |  |  |  | 0.4930 +/- 0.0517 |
| `mlp_h128` | 0.5657 +/- 0.0339 | 0.6617 +/- 0.0220 |  |  |  | 0.6802 +/- 0.1165 |
| `mlp_h64_64` | 0.6451 +/- 0.0394 | 0.7386 +/- 0.0285 |  |  |  | 0.8637 +/- 0.1765 |
| `d18_step2_canonical` | 0.0373 +/- 0.0043 | 0.1029 +/- 0.0108 |  |  | 320.0000 +/- 0.0000 | 3.8737 +/- 0.2888 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4556 +/- 0.0305; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 1.9936 +/- 0.3256 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 2.5619 +/- 0.8469 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 1.7427 +/- 0.0944 |
| `d18_step2_canonical` | 0.0342 +/- 0.0006 | 0.0456 +/- 0.0004 | 0.8783 +/- 0.0061 | 0.9427 +/- 0.0041 | 320.0000 +/- 0.0000 | 3.7974 +/- 0.0514 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0054 +/- 0.0005; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0427 +/- 0.0048; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 1.7193 +/- 0.4311 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 1.9747 +/- 0.5621 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 1.8601 +/- 0.4849 |
| `d18_step2_canonical` | 0.0432 +/- 0.0015 | 0.0525 +/- 0.0009 | 0.7870 +/- 0.0151 | 0.8286 +/- 0.0137 | 320.0000 +/- 0.0000 | 3.4747 +/- 0.2320 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0048 +/- 0.0009; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0074 +/- 0.0107; wins/losses/ties 5/5/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 1.0803 +/- 0.1099 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 1.1741 +/- 0.1079 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 1.1967 +/- 0.1091 |
| `d18_step2_canonical` | 0.0377 +/- 0.0005 | 0.0458 +/- 0.0002 | 0.8683 +/- 0.0050 | 0.9163 +/- 0.0045 | 320.0000 +/- 0.0000 | 3.7774 +/- 0.4012 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0106 +/- 0.0004; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0382 +/- 0.0052; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.5583 +/- 0.0378 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.5102 +/- 0.0418 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.6750 +/- 0.0602 |
| `d18_step2_canonical` | 0.2044 +/- 0.0281 | 0.1671 +/- 0.0128 |  |  | 320.0000 +/- 0.0000 | 4.2655 +/- 0.3275 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0650 +/- 0.0218; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.6271 +/- 0.3675 | 1.4277 +/- 0.1462 |  |  |  | 0.3423 +/- 0.0267 |
| `mlp_h128` | 1.6349 +/- 0.3642 | 1.4399 +/- 0.1466 |  |  |  | 0.3994 +/- 0.0621 |
| `mlp_h64_64` | 1.6000 +/- 0.3571 | 1.4221 +/- 0.1407 |  |  |  | 0.4128 +/- 0.0370 |
| `d18_step2_canonical` | 1.0619 +/- 0.2687 | 0.9476 +/- 0.1025 |  |  | 320.0000 +/- 0.0000 | 2.2823 +/- 0.0701 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5321 +/- 0.0967; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.2157 | 1.0156 +/- 0.0967 |  |  |  | 0.5374 +/- 0.1077 |
| `mlp_h128` | 1.0368 +/- 0.2152 | 1.0187 +/- 0.0969 |  |  |  | 0.5684 +/- 0.1082 |
| `mlp_h64_64` | 1.0051 +/- 0.2125 | 0.9966 +/- 0.0942 |  |  |  | 0.5768 +/- 0.0306 |
| `d18_step2_canonical` | 0.8543 +/- 0.1864 | 0.7668 +/- 0.0809 |  |  | 320.0000 +/- 0.0000 | 3.4789 +/- 0.0645 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1508 +/- 0.0333; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

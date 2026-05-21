# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_basis_0p6.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3952 +/- 0.0203 | 0.5573 +/- 0.0111 |  |  |  | 0.2557 +/- 0.0862 |
| `mlp_h128` | 0.4233 +/- 0.0144 | 0.5731 +/- 0.0089 |  |  |  | 0.2874 +/- 0.0630 |
| `mlp_h64_64` | 0.1811 +/- 0.0157 | 0.3884 +/- 0.0142 |  |  |  | 0.3300 +/- 0.0322 |
| `d18_step2_basis_0p6` | 0.0483 +/- 0.0020 | 0.1490 +/- 0.0016 |  |  | 320.0000 +/- 0.0000 | 1.5673 +/- 0.0863 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1328 +/- 0.0156; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4929 +/- 0.0342 | 0.6412 +/- 0.0189 |  |  |  | 0.1976 +/- 0.0137 |
| `mlp_h128` | 0.5657 +/- 0.0339 | 0.6617 +/- 0.0220 |  |  |  | 0.2166 +/- 0.0191 |
| `mlp_h64_64` | 0.6451 +/- 0.0394 | 0.7386 +/- 0.0285 |  |  |  | 0.2936 +/- 0.0259 |
| `d18_step2_basis_0p6` | 0.0410 +/- 0.0048 | 0.1082 +/- 0.0113 |  |  | 320.0000 +/- 0.0000 | 1.5911 +/- 0.0711 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4518 +/- 0.0301; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 0.7554 +/- 0.0648 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 0.8360 +/- 0.0670 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 0.9901 +/- 0.1008 |
| `d18_step2_basis_0p6` | 0.0359 +/- 0.0006 | 0.0470 +/- 0.0004 | 0.8717 +/- 0.0051 | 0.9358 +/- 0.0035 | 320.0000 +/- 0.0000 | 3.4359 +/- 0.2946 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0037 +/- 0.0005; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0358 +/- 0.0046; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 1.3406 +/- 0.1807 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 1.6535 +/- 0.2505 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 1.8629 +/- 0.3372 |
| `d18_step2_basis_0p6` | 0.0463 +/- 0.0015 | 0.0558 +/- 0.0009 | 0.7657 +/- 0.0159 | 0.8134 +/- 0.0164 | 320.0000 +/- 0.0000 | 7.5348 +/- 0.9028 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0017 +/- 0.0009; wins/losses/ties 7/3/0; best-D18 counts {'d18_step2_basis_0p6': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0226 +/- 0.0120; wins/losses/ties 3/7/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 1.6193 +/- 0.2289 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 1.6129 +/- 0.3716 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 1.6914 +/- 0.2012 |
| `d18_step2_basis_0p6` | 0.0399 +/- 0.0006 | 0.0478 +/- 0.0002 | 0.8507 +/- 0.0061 | 0.9046 +/- 0.0058 | 320.0000 +/- 0.0000 | 9.3122 +/- 1.2572 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0085 +/- 0.0004; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0265 +/- 0.0050; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.7039 +/- 0.1512 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.6353 +/- 0.1767 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.5944 +/- 0.0416 |
| `d18_step2_basis_0p6` | 0.2038 +/- 0.0299 | 0.1689 +/- 0.0129 |  |  | 320.0000 +/- 0.0000 | 5.7998 +/- 0.7115 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0655 +/- 0.0227; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.6271 +/- 0.3675 | 1.4277 +/- 0.1462 |  |  |  | 0.4353 +/- 0.0711 |
| `mlp_h128` | 1.6349 +/- 0.3642 | 1.4399 +/- 0.1466 |  |  |  | 0.4426 +/- 0.0683 |
| `mlp_h64_64` | 1.6000 +/- 0.3571 | 1.4221 +/- 0.1407 |  |  |  | 0.5991 +/- 0.0690 |
| `d18_step2_basis_0p6` | 1.0990 +/- 0.2889 | 0.9805 +/- 0.1080 |  |  | 320.0000 +/- 0.0000 | 5.3975 +/- 0.8165 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4950 +/- 0.0785; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.2157 | 1.0156 +/- 0.0967 |  |  |  | 0.6952 +/- 0.1212 |
| `mlp_h128` | 1.0368 +/- 0.2152 | 1.0187 +/- 0.0969 |  |  |  | 0.7015 +/- 0.1414 |
| `mlp_h64_64` | 1.0051 +/- 0.2125 | 0.9966 +/- 0.0942 |  |  |  | 0.7456 +/- 0.0907 |
| `d18_step2_basis_0p6` | 0.8573 +/- 0.1863 | 0.7719 +/- 0.0810 |  |  | 320.0000 +/- 0.0000 | 7.4339 +/- 0.9301 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1478 +/- 0.0341; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

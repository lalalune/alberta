# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_basis_0p6.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3952 +/- 0.0203 | 0.5573 +/- 0.0111 |  |  |  | 0.3126 +/- 0.0870 |
| `mlp_h128` | 0.4233 +/- 0.0144 | 0.5731 +/- 0.0089 |  |  |  | 0.3008 +/- 0.0702 |
| `mlp_h64_64` | 0.1811 +/- 0.0157 | 0.3884 +/- 0.0142 |  |  |  | 0.2871 +/- 0.0353 |
| `d18_step2_basis_0p6` | 0.0502 +/- 0.0018 | 0.1294 +/- 0.0013 |  |  | 320.0000 +/- 0.0000 | 1.5544 +/- 0.0920 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1309 +/- 0.0155; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4929 +/- 0.0342 | 0.6412 +/- 0.0189 |  |  |  | 0.2066 +/- 0.0132 |
| `mlp_h128` | 0.5657 +/- 0.0339 | 0.6617 +/- 0.0220 |  |  |  | 0.2167 +/- 0.0275 |
| `mlp_h64_64` | 0.6451 +/- 0.0394 | 0.7386 +/- 0.0285 |  |  |  | 0.2728 +/- 0.0183 |
| `d18_step2_basis_0p6` | 0.0421 +/- 0.0049 | 0.1097 +/- 0.0114 |  |  | 320.0000 +/- 0.0000 | 1.5737 +/- 0.0736 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4508 +/- 0.0301; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 0.7992 +/- 0.0792 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 0.9363 +/- 0.0892 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 0.9519 +/- 0.1070 |
| `d18_step2_basis_0p6` | 0.0350 +/- 0.0006 | 0.0467 +/- 0.0003 | 0.8803 +/- 0.0052 | 0.9382 +/- 0.0035 | 320.0000 +/- 0.0000 | 3.4573 +/- 0.2696 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0046 +/- 0.0005; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0382 +/- 0.0044; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 1.7181 +/- 0.3114 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 1.8562 +/- 0.3573 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 1.6181 +/- 0.2350 |
| `d18_step2_basis_0p6` | 0.0458 +/- 0.0015 | 0.0554 +/- 0.0009 | 0.7750 +/- 0.0147 | 0.8223 +/- 0.0158 | 320.0000 +/- 0.0000 | 7.3250 +/- 0.8480 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0023 +/- 0.0008; wins/losses/ties 7/3/0; best-D18 counts {'d18_step2_basis_0p6': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0137 +/- 0.0114; wins/losses/ties 4/6/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 1.7848 +/- 0.3094 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 1.4154 +/- 0.2278 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 2.0270 +/- 0.3876 |
| `d18_step2_basis_0p6` | 0.0393 +/- 0.0005 | 0.0475 +/- 0.0002 | 0.8580 +/- 0.0055 | 0.9095 +/- 0.0057 | 320.0000 +/- 0.0000 | 9.0648 +/- 1.1852 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0090 +/- 0.0004; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0314 +/- 0.0048; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.5217 +/- 0.0889 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.5609 +/- 0.1071 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.7342 +/- 0.0956 |
| `d18_step2_basis_0p6` | 0.2049 +/- 0.0300 | 0.1688 +/- 0.0131 |  |  | 320.0000 +/- 0.0000 | 5.5981 +/- 0.7000 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0645 +/- 0.0220; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.6271 +/- 0.3675 | 1.4277 +/- 0.1462 |  |  |  | 0.4627 +/- 0.0679 |
| `mlp_h128` | 1.6349 +/- 0.3642 | 1.4399 +/- 0.1466 |  |  |  | 0.5652 +/- 0.1005 |
| `mlp_h64_64` | 1.6000 +/- 0.3571 | 1.4221 +/- 0.1407 |  |  |  | 0.7110 +/- 0.1064 |
| `d18_step2_basis_0p6` | 1.0477 +/- 0.2687 | 0.9312 +/- 0.1019 |  |  | 320.0000 +/- 0.0000 | 5.2809 +/- 0.8345 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5463 +/- 0.0959; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.2157 | 1.0156 +/- 0.0967 |  |  |  | 0.7220 +/- 0.1300 |
| `mlp_h128` | 1.0368 +/- 0.2152 | 1.0187 +/- 0.0969 |  |  |  | 0.7163 +/- 0.1218 |
| `mlp_h64_64` | 1.0051 +/- 0.2125 | 0.9966 +/- 0.0942 |  |  |  | 0.7286 +/- 0.1104 |
| `d18_step2_basis_0p6` | 0.8606 +/- 0.1867 | 0.7762 +/- 0.0815 |  |  | 320.0000 +/- 0.0000 | 7.2986 +/- 0.9398 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1445 +/- 0.0340; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

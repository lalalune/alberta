# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3952 +/- 0.0203 | 0.5573 +/- 0.0111 |  |  |  | 0.8271 +/- 0.2303 |
| `mlp_h128` | 0.4233 +/- 0.0144 | 0.5731 +/- 0.0089 |  |  |  | 0.6777 +/- 0.1487 |
| `mlp_h64_64` | 0.1811 +/- 0.0157 | 0.3884 +/- 0.0142 |  |  |  | 0.6896 +/- 0.0728 |
| `d18_step2_canonical` | 0.0424 +/- 0.0016 | 0.1468 +/- 0.0017 |  |  | 320.0000 +/- 0.0000 | 4.9111 +/- 0.7007 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1387 +/- 0.0156; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4929 +/- 0.0342 | 0.6412 +/- 0.0189 |  |  |  | 1.1096 +/- 0.3010 |
| `mlp_h128` | 0.5657 +/- 0.0339 | 0.6617 +/- 0.0220 |  |  |  | 0.7185 +/- 0.0827 |
| `mlp_h64_64` | 0.6451 +/- 0.0394 | 0.7386 +/- 0.0285 |  |  |  | 1.1129 +/- 0.1766 |
| `d18_step2_canonical` | 0.0367 +/- 0.0043 | 0.1020 +/- 0.0108 |  |  | 320.0000 +/- 0.0000 | 4.3982 +/- 0.4670 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4561 +/- 0.0305; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 1.6139 +/- 0.1949 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 1.9785 +/- 0.2772 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 1.4569 +/- 0.0989 |
| `d18_step2_canonical` | 0.0349 +/- 0.0006 | 0.0458 +/- 0.0004 | 0.8730 +/- 0.0060 | 0.9410 +/- 0.0037 | 320.0000 +/- 0.0000 | 3.8147 +/- 0.0400 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0047 +/- 0.0005; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0410 +/- 0.0046; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 1.7679 +/- 0.4063 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 2.0402 +/- 0.5173 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 1.9584 +/- 0.5688 |
| `d18_step2_canonical` | 0.0437 +/- 0.0015 | 0.0528 +/- 0.0009 | 0.7803 +/- 0.0145 | 0.8237 +/- 0.0145 | 320.0000 +/- 0.0000 | 3.4041 +/- 0.2105 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0044 +/- 0.0009; wins/losses/ties 9/1/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0122 +/- 0.0110; wins/losses/ties 4/6/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 0.9731 +/- 0.1420 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 1.0288 +/- 0.0898 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 1.1428 +/- 0.1465 |
| `d18_step2_canonical` | 0.0382 +/- 0.0005 | 0.0461 +/- 0.0002 | 0.8680 +/- 0.0059 | 0.9122 +/- 0.0055 | 320.0000 +/- 0.0000 | 3.7325 +/- 0.4070 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0102 +/- 0.0004; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0341 +/- 0.0056; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.6474 +/- 0.0881 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.5659 +/- 0.0903 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.6508 +/- 0.0551 |
| `d18_step2_canonical` | 0.2020 +/- 0.0281 | 0.1664 +/- 0.0126 |  |  | 320.0000 +/- 0.0000 | 4.2247 +/- 0.2448 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0673 +/- 0.0224; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.6271 +/- 0.3675 | 1.4277 +/- 0.1462 |  |  |  | 0.3459 +/- 0.0278 |
| `mlp_h128` | 1.6349 +/- 0.3642 | 1.4399 +/- 0.1466 |  |  |  | 0.3306 +/- 0.0500 |
| `mlp_h64_64` | 1.6000 +/- 0.3571 | 1.4221 +/- 0.1407 |  |  |  | 0.3842 +/- 0.0308 |
| `d18_step2_canonical` | 1.1087 +/- 0.2880 | 0.9942 +/- 0.1084 |  |  | 320.0000 +/- 0.0000 | 2.3844 +/- 0.1102 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4853 +/- 0.0798; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.2157 | 1.0156 +/- 0.0967 |  |  |  | 0.5789 +/- 0.0997 |
| `mlp_h128` | 1.0368 +/- 0.2152 | 1.0187 +/- 0.0969 |  |  |  | 0.6288 +/- 0.0907 |
| `mlp_h64_64` | 1.0051 +/- 0.2125 | 0.9966 +/- 0.0942 |  |  |  | 0.6470 +/- 0.0492 |
| `d18_step2_canonical` | 0.8511 +/- 0.1862 | 0.7625 +/- 0.0806 |  |  | 320.0000 +/- 0.0000 | 3.4939 +/- 0.0551 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1540 +/- 0.0336; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_simplex.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3952 +/- 0.0203 | 0.5573 +/- 0.0111 |  |  |  | 0.1242 +/- 0.0066 |
| `mlp_h128` | 0.4233 +/- 0.0144 | 0.5731 +/- 0.0089 |  |  |  | 0.1271 +/- 0.0068 |
| `mlp_h64_64` | 0.1811 +/- 0.0157 | 0.3884 +/- 0.0142 |  |  |  | 0.2083 +/- 0.0210 |
| `d18_step2_simplex` | 0.0373 +/- 0.0006 | 0.0871 +/- 0.0007 |  |  | 320.0000 +/- 0.0000 | 1.5068 +/- 0.1014 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1438 +/- 0.0158; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4929 +/- 0.0342 | 0.6412 +/- 0.0189 |  |  |  | 0.2402 +/- 0.0315 |
| `mlp_h128` | 0.5657 +/- 0.0339 | 0.6617 +/- 0.0220 |  |  |  | 0.1905 +/- 0.0188 |
| `mlp_h64_64` | 0.6451 +/- 0.0394 | 0.7386 +/- 0.0285 |  |  |  | 0.2825 +/- 0.0237 |
| `d18_step2_simplex` | 0.0407 +/- 0.0044 | 0.1078 +/- 0.0109 |  |  | 320.0000 +/- 0.0000 | 1.9764 +/- 0.1161 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4522 +/- 0.0303; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0707 +/- 0.0041 | 0.1097 +/- 0.0032 |  |  |  | 0.2725 +/- 0.0921 |
| `mlp_h128` | 0.0872 +/- 0.0053 | 0.1191 +/- 0.0029 |  |  |  | 0.2822 +/- 0.1063 |
| `mlp_h64_64` | 0.0952 +/- 0.0076 | 0.1309 +/- 0.0031 |  |  |  | 0.2964 +/- 0.0646 |
| `d18_step2_simplex` | 0.0200 +/- 0.0011 | 0.0451 +/- 0.0009 |  |  | 320.0000 +/- 0.0000 | 1.7290 +/- 0.0933 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0507 +/- 0.0037; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9111 +/- 0.0868 | 1.0747 +/- 0.0434 |  |  |  | 0.1069 +/- 0.0011 |
| `mlp_h128` | 1.0331 +/- 0.0969 | 1.1401 +/- 0.0493 |  |  |  | 0.1096 +/- 0.0016 |
| `mlp_h64_64` | 1.0520 +/- 0.0993 | 1.1557 +/- 0.0536 |  |  |  | 0.1507 +/- 0.0039 |
| `d18_step2_simplex` | 0.0935 +/- 0.0156 | 0.3175 +/- 0.0285 |  |  | 320.0000 +/- 0.0000 | 1.1776 +/- 0.0052 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.8172 +/- 0.0756; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0938 +/- 0.0115 | 0.1103 +/- 0.0080 |  |  |  | 0.1267 +/- 0.0022 |
| `mlp_h128` | 0.0981 +/- 0.0112 | 0.1134 +/- 0.0076 |  |  |  | 0.1274 +/- 0.0017 |
| `mlp_h64_64` | 0.1150 +/- 0.0110 | 0.1292 +/- 0.0077 |  |  |  | 0.1740 +/- 0.0086 |
| `d18_step2_simplex` | 0.0319 +/- 0.0074 | 0.0504 +/- 0.0053 |  |  | 320.0000 +/- 0.0000 | 1.1959 +/- 0.0146 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0618 +/- 0.0052; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0086 +/- 0.0515 | 1.1263 +/- 0.0416 |  |  |  | 0.2133 +/- 0.0348 |
| `mlp_h128` | 1.0193 +/- 0.0526 | 1.1366 +/- 0.0417 |  |  |  | 0.1853 +/- 0.0214 |
| `mlp_h64_64` | 0.7461 +/- 0.0462 | 0.9548 +/- 0.0445 |  |  |  | 0.2417 +/- 0.0177 |
| `d18_step2_simplex` | 0.0563 +/- 0.0049 | 0.2607 +/- 0.0236 |  |  | 320.0000 +/- 0.0000 | 1.8356 +/- 0.2173 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.6898 +/- 0.0424; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.7874 +/- 0.1408 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.8450 +/- 0.1322 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 1.0804 +/- 0.1709 |
| `d18_step2_simplex` | 0.0015 +/- 0.0001 | 0.0020 +/- 0.0001 | 0.9927 +/- 0.0004 | 0.1006 +/- 0.0004 | 320.0000 +/- 0.0000 | 4.1117 +/- 0.4683 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0014 +/- 0.0001; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0353 +/- 0.0119; wins/losses/ties 1/9/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0315 +/- 0.0005 | 0.0452 +/- 0.0004 | 0.9123 +/- 0.0053 | 0.9148 +/- 0.0041 |  | 0.5129 +/- 0.0637 |
| `mlp_h128` | 0.0306 +/- 0.0007 | 0.0438 +/- 0.0003 | 0.9257 +/- 0.0056 | 0.9306 +/- 0.0045 |  | 0.5488 +/- 0.0621 |
| `mlp_h64_64` | 0.0325 +/- 0.0006 | 0.0486 +/- 0.0004 | 0.8900 +/- 0.0058 | 0.9058 +/- 0.0061 |  | 0.5917 +/- 0.0868 |
| `d18_step2_simplex` | 0.0080 +/- 0.0006 | 0.0198 +/- 0.0004 | 0.9600 +/- 0.0030 | 0.9647 +/- 0.0039 | 320.0000 +/- 0.0000 | 4.1013 +/- 0.6717 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0224 +/- 0.0009; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0338 +/- 0.0037; wins/losses/ties 10/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 0.8084 +/- 0.0520 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 0.8727 +/- 0.0501 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 0.9717 +/- 0.0734 |
| `d18_step2_simplex` | 0.0263 +/- 0.0010 | 0.0464 +/- 0.0009 | 0.8687 +/- 0.0051 | 0.9458 +/- 0.0043 | 320.0000 +/- 0.0000 | 3.5556 +/- 0.1740 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0133 +/- 0.0007; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0458 +/- 0.0061; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.7750 +/- 0.0401 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.7848 +/- 0.0411 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.9345 +/- 0.0616 |
| `d18_step2_simplex` | 0.0397 +/- 0.0029 | 0.0555 +/- 0.0019 | 0.8013 +/- 0.0146 | 0.8442 +/- 0.0125 | 320.0000 +/- 0.0000 | 4.7005 +/- 0.3087 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0083 +/- 0.0022; wins/losses/ties 9/1/0; best-D18 counts {'d18_step2_simplex': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0082 +/- 0.0094; wins/losses/ties 6/4/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 0.9518 +/- 0.1015 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 0.7672 +/- 0.0594 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 0.9469 +/- 0.0848 |
| `d18_step2_simplex` | 0.0253 +/- 0.0008 | 0.0414 +/- 0.0004 | 0.8737 +/- 0.0041 | 0.9223 +/- 0.0046 | 320.0000 +/- 0.0000 | 6.2692 +/- 0.7938 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0231 +/- 0.0006; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0442 +/- 0.0058; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.2963 +/- 0.0818 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.2555 +/- 0.0512 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.2303 +/- 0.0173 |
| `d18_step2_simplex` | 0.2244 +/- 0.0285 | 0.1764 +/- 0.0139 |  |  | 320.0000 +/- 0.0000 | 1.5331 +/- 0.1092 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0449 +/- 0.0187; wins/losses/ties 9/1/0; best-D18 counts {'d18_step2_simplex': 10}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.6271 +/- 0.3675 | 1.4277 +/- 0.1462 |  |  |  | 0.2112 +/- 0.0165 |
| `mlp_h128` | 1.6349 +/- 0.3642 | 1.4399 +/- 0.1466 |  |  |  | 0.2511 +/- 0.0357 |
| `mlp_h64_64` | 1.6000 +/- 0.3571 | 1.4221 +/- 0.1407 |  |  |  | 0.2978 +/- 0.0306 |
| `d18_step2_simplex` | 1.0595 +/- 0.2397 | 0.8799 +/- 0.0942 |  |  | 320.0000 +/- 0.0000 | 2.0209 +/- 0.2524 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5345 +/- 0.1240; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.2157 | 1.0156 +/- 0.0967 |  |  |  | 0.3131 +/- 0.0879 |
| `mlp_h128` | 1.0368 +/- 0.2152 | 1.0187 +/- 0.0969 |  |  |  | 0.3010 +/- 0.0734 |
| `mlp_h64_64` | 1.0051 +/- 0.2125 | 0.9966 +/- 0.0942 |  |  |  | 0.2853 +/- 0.0161 |
| `d18_step2_simplex` | 0.8565 +/- 0.1842 | 0.7658 +/- 0.0807 |  |  | 320.0000 +/- 0.0000 | 2.1713 +/- 0.0966 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1485 +/- 0.0353; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.

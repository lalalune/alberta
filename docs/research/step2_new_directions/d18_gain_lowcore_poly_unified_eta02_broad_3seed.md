# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_lowcore_poly_unified.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3511 +/- 0.0292 | 0.5451 +/- 0.0160 |  |  |  | 1.4986 +/- 1.0860 |
| `mlp_h128` | 0.4174 +/- 0.0143 | 0.5811 +/- 0.0160 |  |  |  | 1.6778 +/- 1.1613 |
| `mlp_h64_64` | 0.1569 +/- 0.0264 | 0.3911 +/- 0.0292 |  |  |  | 1.6196 +/- 1.0872 |
| `d18_gain_lowcore_poly_unified` | 0.0481 +/- 0.0011 | 0.0884 +/- 0.0017 |  |  | 320.0000 +/- 0.0000 | 19.8866 +/- 2.9310 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1087 +/- 0.0270; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  | 1.0939 +/- 0.2881 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  | 1.1412 +/- 0.1994 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  | 1.5242 +/- 0.1335 |
| `d18_gain_lowcore_poly_unified` | 0.0247 +/- 0.0040 | 0.1109 +/- 0.0127 |  |  | 320.0000 +/- 0.0000 | 29.3599 +/- 7.6328 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.4036 +/- 0.0215; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 8.0196 +/- 2.1728 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 6.3392 +/- 0.7766 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 5.5399 +/- 0.8096 |
| `d18_gain_lowcore_poly_unified` | 0.0632 +/- 0.0020 | 0.0635 +/- 0.0012 | 0.5900 +/- 0.0158 | 0.7372 +/- 0.0569 | 320.0000 +/- 0.0000 | 50.1500 +/- 4.1285 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0251 +/- 0.0021; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.1701 +/- 0.0566; wins/losses/ties 0/3/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 7.1268 +/- 2.3488 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 4.7938 +/- 1.6008 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 5.7326 +/- 1.1688 |
| `d18_gain_lowcore_poly_unified` | 0.0578 +/- 0.0013 | 0.0629 +/- 0.0019 | 0.6544 +/- 0.0154 | 0.6821 +/- 0.0223 | 320.0000 +/- 0.0000 | 28.8884 +/- 4.1954 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0099 +/- 0.0012; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.1379 +/- 0.0303; wins/losses/ties 0/3/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 |  | 2.7277 +/- 0.1627 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 |  | 2.2765 +/- 0.5258 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 |  | 2.4394 +/- 0.0753 |
| `d18_gain_lowcore_poly_unified` | 0.0636 +/- 0.0034 | 0.0619 +/- 0.0008 | 0.7067 +/- 0.0139 | 0.7922 +/- 0.0075 | 320.0000 +/- 0.0000 | 25.0395 +/- 4.3582 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0144 +/- 0.0029; wins/losses/ties 0/3/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0891 +/- 0.0134; wins/losses/ties 0/3/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.8233 +/- 1.0739 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.2216 +/- 0.5943 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.7634 +/- 0.0713 |
| `d18_gain_lowcore_poly_unified` | 0.2757 +/- 0.1083 | 0.1636 +/- 0.0327 |  |  | 320.0000 +/- 0.0000 | 9.9391 +/- 1.0003 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0040 +/- 0.0164; wins/losses/ties 1/2/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.6755 +/- 0.1370 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.7029 +/- 0.1435 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.9301 +/- 0.2527 |
| `d18_gain_lowcore_poly_unified` | 1.0554 +/- 0.2618 | 1.0098 +/- 0.2183 |  |  | 320.0000 +/- 0.0000 | 10.6508 +/- 3.2665 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0930 +/- 0.1146; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 3.3380 +/- 1.5710 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 2.2820 +/- 1.5371 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 1.0222 +/- 0.1485 |
| `d18_gain_lowcore_poly_unified` | 0.6564 +/- 0.3797 | 0.5890 +/- 0.1159 |  |  | 320.0000 +/- 0.0000 | 19.1089 +/- 0.9145 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2910 +/- 0.2648; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 3}.

# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.9229 +/- 0.4119 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.9293 +/- 0.2647 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.8933 +/- 0.2112 |
| `d18_step2_canonical` | 0.0165 +/- 0.0052 | 0.0521 +/- 0.0128 |  |  | 160.0000 +/- 0.0000 | 7.1224 +/- 0.4082 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0567 +/- 0.0066; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 1.7200 +/- 0.2605 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 1.7882 +/- 0.2926 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 1.8709 +/- 0.1651 |
| `d18_step2_canonical` | 0.0048 +/- 0.0000 | 0.0042 +/- 0.0000 | 0.9856 +/- 0.0011 | 0.2115 +/- 0.0253 | 160.0000 +/- 0.0000 | 9.8920 +/- 0.4908 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0018 +/- 0.0001; wins/losses/ties 0/3/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0458 +/- 0.0134; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 4.0130 +/- 1.3487 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 3.3575 +/- 0.8418 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 4.2125 +/- 1.6607 |
| `d18_step2_canonical` | 0.0464 +/- 0.0021 | 0.0540 +/- 0.0011 | 0.7744 +/- 0.0225 | 0.8033 +/- 0.0215 | 160.0000 +/- 0.0000 | 12.6852 +/- 0.9305 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0015 +/- 0.0014; wins/losses/ties 2/1/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0167 +/- 0.0093; wins/losses/ties 0/3/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.3112 +/- 0.8676 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 2.2870 +/- 1.7476 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.6977 +/- 0.9728 |
| `d18_step2_canonical` | 0.2446 +/- 0.0827 | 0.1559 +/- 0.0272 |  |  | 160.0000 +/- 0.0000 | 12.7500 +/- 7.0905 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0271 +/- 0.0128; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.5743 +/- 0.1633 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.6609 +/- 0.1753 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.8186 +/- 0.1678 |
| `d18_step2_canonical` | 0.9025 +/- 0.2205 | 0.8744 +/- 0.1676 |  |  | 160.0000 +/- 0.0000 | 3.3927 +/- 0.8371 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2460 +/- 0.0743; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.9234 +/- 0.2597 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.9359 +/- 0.5053 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.5758 +/- 0.0422 |
| `d18_step2_canonical` | 0.8148 +/- 0.5414 | 0.8144 +/- 0.2062 |  |  | 160.0000 +/- 0.0000 | 7.9063 +/- 1.9207 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1326 +/- 0.1031; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

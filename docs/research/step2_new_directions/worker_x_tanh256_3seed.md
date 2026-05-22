# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.5387 +/- 0.2016 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.5463 +/- 0.1498 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.4295 +/- 0.0877 |
| `d18_step2_canonical` | 0.0179 +/- 0.0064 | 0.0514 +/- 0.0119 |  |  | 320.0000 +/- 0.0000 | 3.6839 +/- 1.2060 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0553 +/- 0.0055; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 1.0072 +/- 0.2987 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 0.7759 +/- 0.1603 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 0.8654 +/- 0.2043 |
| `d18_step2_canonical` | 0.0046 +/- 0.0001 | 0.0041 +/- 0.0000 | 0.9889 +/- 0.0011 | 0.2177 +/- 0.0213 | 320.0000 +/- 0.0000 | 3.6593 +/- 0.3430 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0016 +/- 0.0001; wins/losses/ties 0/3/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0519 +/- 0.0132; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 6.7522 +/- 5.3358 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 5.0816 +/- 3.8712 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 3.1754 +/- 1.4839 |
| `d18_step2_canonical` | 0.0449 +/- 0.0017 | 0.0528 +/- 0.0012 | 0.7744 +/- 0.0231 | 0.8287 +/- 0.0221 | 320.0000 +/- 0.0000 | 20.5648 +/- 11.3202 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0029 +/- 0.0010; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0087 +/- 0.0125; wins/losses/ties 2/1/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.6607 +/- 0.2279 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.6720 +/- 0.2725 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.5568 +/- 0.0214 |
| `d18_step2_canonical` | 0.2424 +/- 0.0793 | 0.1526 +/- 0.0265 |  |  | 320.0000 +/- 0.0000 | 3.5881 +/- 0.2373 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0293 +/- 0.0163; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.4456 +/- 0.0486 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.6205 +/- 0.2014 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.5304 +/- 0.0548 |
| `d18_step2_canonical` | 0.8852 +/- 0.2025 | 0.8903 +/- 0.1772 |  |  | 320.0000 +/- 0.0000 | 9.6087 +/- 0.5141 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2632 +/- 0.0771; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.8904 +/- 0.4568 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.9590 +/- 0.5187 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.5538 +/- 0.0547 |
| `d18_step2_canonical` | 0.7963 +/- 0.5188 | 0.8067 +/- 0.2064 |  |  | 320.0000 +/- 0.0000 | 6.8439 +/- 1.0964 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1511 +/- 0.1257; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

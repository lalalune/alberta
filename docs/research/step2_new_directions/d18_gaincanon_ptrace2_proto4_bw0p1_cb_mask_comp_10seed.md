# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.4745 +/- 0.0908 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.4390 +/- 0.0872 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.4211 +/- 0.0173 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0332 +/- 0.0015 | 0.0225 +/- 0.0004 | 0.8340 +/- 0.0075 | 0.8419 +/- 0.0060 | 320.0000 +/- 0.0000 | 1.9950 +/- 0.0126 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0303 +/- 0.0015; wins/losses/ties 0/10/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.7061 +/- 0.0152; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.3431 +/- 0.0138 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.3879 +/- 0.0104 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.3864 +/- 0.0208 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0311 +/- 0.0021 | 0.0429 +/- 0.0015 | 0.8447 +/- 0.0106 | 0.8679 +/- 0.0073 | 320.0000 +/- 0.0000 | 2.0048 +/- 0.0183 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0169 +/- 0.0011; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0319 +/- 0.0074; wins/losses/ties 9/1/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.1851 +/- 0.0332 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.1855 +/- 0.0338 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.2039 +/- 0.0049 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2064 +/- 0.0277 | 0.1683 +/- 0.0129 |  |  | 320.0000 +/- 0.0000 | 1.4656 +/- 0.0804 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0629 +/- 0.0210; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.

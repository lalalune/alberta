# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.5881 +/- 0.0448 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.6285 +/- 0.0415 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.5811 +/- 0.0233 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0441 +/- 0.0015 | 0.0534 +/- 0.0009 | 0.7847 +/- 0.0146 | 0.8280 +/- 0.0147 | 320.0000 +/- 0.0000 | 2.3125 +/- 0.0974 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0039 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0080 +/- 0.0112; wins/losses/ties 4/6/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.3559 +/- 0.1261 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.3513 +/- 0.0953 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.3992 +/- 0.0575 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2061 +/- 0.0272 | 0.1682 +/- 0.0129 |  |  | 320.0000 +/- 0.0000 | 1.6486 +/- 0.0284 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0632 +/- 0.0208; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.

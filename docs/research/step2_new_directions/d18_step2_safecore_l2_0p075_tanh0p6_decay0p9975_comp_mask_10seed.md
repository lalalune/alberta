# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.4209 +/- 0.0255 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.5547 +/- 0.0411 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.6015 +/- 0.0356 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0436 +/- 0.0015 | 0.0529 +/- 0.0009 | 0.7890 +/- 0.0143 | 0.8308 +/- 0.0140 | 320.0000 +/- 0.0000 | 2.4970 +/- 0.1277 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0044 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0052 +/- 0.0107; wins/losses/ties 4/6/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.4112 +/- 0.1259 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.3774 +/- 0.1079 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.3587 +/- 0.0541 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2065 +/- 0.0276 | 0.1682 +/- 0.0129 |  |  | 320.0000 +/- 0.0000 | 1.6010 +/- 0.0370 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0628 +/- 0.0210; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.

# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p25_basis_0p6, d18_gain_lowcore_unified, d18_gain_lowcore_poly_unified, d18_gain_safe_digits.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 4.3173 +/- 2.5114 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 3.9464 +/- 2.8789 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 2.6791 +/- 0.8782 |
| `d18_core_0p25_basis_0p6` | 1.0380 +/- 0.6272 | 1.1072 +/- 0.2764 |  |  | 320.0000 +/- 0.0000 | 15.7859 +/- 2.8738 |
| `d18_gain_lowcore_unified` | 0.8933 +/- 0.6122 | 0.9667 +/- 0.2365 |  |  | 320.0000 +/- 0.0000 | 23.7392 +/- 5.1644 |
| `d18_gain_lowcore_poly_unified` | 0.6571 +/- 0.3797 | 0.5882 +/- 0.1158 |  |  | 320.0000 +/- 0.0000 | 28.3434 +/- 7.3873 |
| `d18_gain_safe_digits` | 0.8957 +/- 0.6105 | 0.9717 +/- 0.2370 |  |  | 320.0000 +/- 0.0000 | 35.1106 +/- 16.2623 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2918 +/- 0.2641; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 2, 'd18_gain_lowcore_unified': 1}.

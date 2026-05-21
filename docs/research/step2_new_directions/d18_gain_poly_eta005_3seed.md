# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p25_basis_0p6, d18_gain_lowcore_unified, d18_gain_lowcore_poly_unified, d18_gain_safe_digits.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 4.5075 +/- 2.4131 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 4.4713 +/- 2.4180 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 3.3869 +/- 1.0598 |
| `d18_core_0p25_basis_0p6` | 1.0380 +/- 0.6272 | 1.1072 +/- 0.2764 |  |  | 320.0000 +/- 0.0000 | 15.5316 +/- 2.5830 |
| `d18_gain_lowcore_unified` | 0.8660 +/- 0.6012 | 0.9483 +/- 0.2287 |  |  | 320.0000 +/- 0.0000 | 22.5064 +/- 5.5421 |
| `d18_gain_lowcore_poly_unified` | 0.6784 +/- 0.3906 | 0.6726 +/- 0.1496 |  |  | 320.0000 +/- 0.0000 | 29.1835 +/- 6.1982 |
| `d18_gain_safe_digits` | 0.8650 +/- 0.5949 | 0.9481 +/- 0.2288 |  |  | 320.0000 +/- 0.0000 | 32.3280 +/- 15.7216 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2844 +/- 0.2462; wins/losses/ties 3/0/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 1, 'd18_gain_lowcore_unified': 2}.

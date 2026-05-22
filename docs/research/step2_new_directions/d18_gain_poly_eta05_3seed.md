# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p25_basis_0p6, d18_gain_lowcore_unified, d18_gain_lowcore_poly_unified, d18_gain_safe_digits.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 3.8170 +/- 2.6178 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 4.1653 +/- 2.2723 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 2.9837 +/- 1.2000 |
| `d18_core_0p25_basis_0p6` | 1.0380 +/- 0.6272 | 1.1072 +/- 0.2764 |  |  | 320.0000 +/- 0.0000 | 15.6265 +/- 2.6409 |
| `d18_gain_lowcore_unified` | 0.9508 +/- 0.6398 | 1.0396 +/- 0.2620 |  |  | 320.0000 +/- 0.0000 | 22.7257 +/- 5.6960 |
| `d18_gain_lowcore_poly_unified` | 0.7254 +/- 0.4195 | 0.6272 +/- 0.1269 |  |  | 320.0000 +/- 0.0000 | 27.7529 +/- 6.0178 |
| `d18_gain_safe_digits` | 0.9458 +/- 0.6322 | 1.0410 +/- 0.2597 |  |  | 320.0000 +/- 0.0000 | 31.3320 +/- 13.3375 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2291 +/- 0.2221; wins/losses/ties 2/1/0; best-D18 counts {'d18_gain_lowcore_poly_unified': 2, 'd18_gain_safe_digits': 1}.

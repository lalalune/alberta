# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p5_basis_0p4_poly_0p25, d18_core_0p5_basis_0p4_poly_0p4_decay, d18_core_0p5_basis_0p4_poly_0p6_decay.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.1176 +/- 0.7116 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.3082 +/- 0.6582 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.8928 +/- 0.2240 |
| `d18_core_0p5_basis_0p4_poly_0p25` | 1.2109 +/- 0.4351 | 1.1730 +/- 0.2971 |  |  | 320.0000 +/- 0.0000 | 3.5178 +/- 0.1787 |
| `d18_core_0p5_basis_0p4_poly_0p4_decay` | 1.5761 +/- 0.4205 | 1.3724 +/- 0.3367 |  |  | 320.0000 +/- 0.0000 | 4.0627 +/- 0.3781 |
| `d18_core_0p5_basis_0p4_poly_0p6_decay` | 2.6962 +/- 0.4858 | 2.0036 +/- 0.4171 |  |  | 320.0000 +/- 0.0000 | 4.2357 +/- 0.2418 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.2635 +/- 0.2280; wins/losses/ties 1/2/0; best-D18 counts {'d18_core_0p5_basis_0p4_poly_0p25': 3}.

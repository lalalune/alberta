# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p25_basis_0p6_poly_rls_0p1, d18_core_0p25_basis_0p6_poly_rls_0p15, d18_core_0p5_basis_0p4_poly_rls_0p25.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.3457 +/- 0.9364 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.1425 +/- 0.7630 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 1.2044 +/- 0.6366 |
| `d18_core_0p25_basis_0p6_poly_rls_0p1` | 164969.3262 +/- 164341.8374 | 91635.1977 +/- 91382.2378 |  |  | 320.0000 +/- 0.0000 | 6.3069 +/- 2.1187 |
| `d18_core_0p25_basis_0p6_poly_rls_0p15` | 865248.1962 +/- 862204.5149 | 471481.2688 +/- 470180.2612 |  |  | 320.0000 +/- 0.0000 | 6.7076 +/- 1.1151 |
| `d18_core_0p5_basis_0p4_poly_rls_0p25` | 6945632.9142 +/- 6941378.2139 | 3740251.8004 +/- 3734699.2970 |  |  | 320.0000 +/- 0.0000 | 5.7488 +/- 0.3677 |

`final_window_mse` best-D18-vs-best-MLP diff: -164968.3787 +/- 164342.1539; wins/losses/ties 0/3/0; best-D18 counts {'d18_core_0p25_basis_0p6_poly_rls_0p1': 3}.

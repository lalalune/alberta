# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p25_basis_0p6_poly_rls_0p1, d18_core_0p25_basis_0p6_poly_rls_0p15, d18_core_0p5_basis_0p4_poly_rls_0p25.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.2408 +/- 0.7614 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.1389 +/- 0.7164 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.9110 +/- 0.2441 |
| `d18_core_0p25_basis_0p6_poly_rls_0p1` | 394.3369 +/- 168.1572 | 150.0895 +/- 70.3234 |  |  | 320.0000 +/- 0.0000 | 7.2789 +/- 2.1732 |
| `d18_core_0p25_basis_0p6_poly_rls_0p15` | 1658.2926 +/- 829.3258 | 549.0203 +/- 217.7787 |  |  | 320.0000 +/- 0.0000 | 5.0203 +/- 0.6598 |
| `d18_core_0p5_basis_0p4_poly_rls_0p25` | 9132.5184 +/- 5137.5849 | 2811.5464 +/- 1193.3207 |  |  | 320.0000 +/- 0.0000 | 7.0035 +/- 0.8532 |

`final_window_mse` best-D18-vs-best-MLP diff: -393.3895 +/- 168.1569; wins/losses/ties 0/3/0; best-D18 counts {'d18_core_0p25_basis_0p6_poly_rls_0p1': 3}.

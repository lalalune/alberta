# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p25_basis_0p6_poly_rls_0p1, d18_core_0p25_basis_0p6_poly_rls_0p15, d18_core_0p5_basis_0p4_poly_rls_0p25.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.3213 +/- 0.8739 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.3011 +/- 0.6849 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.8216 +/- 0.2859 |
| `d18_core_0p25_basis_0p6_poly_rls_0p1` | 117.4572 +/- 116.1834 | 30.0555 +/- 28.8876 |  |  | 320.0000 +/- 0.0000 | 7.0746 +/- 2.7282 |
| `d18_core_0p25_basis_0p6_poly_rls_0p15` | 362.3018 +/- 359.8474 | 91.2300 +/- 89.8064 |  |  | 320.0000 +/- 0.0000 | 6.4821 +/- 0.5227 |
| `d18_core_0p5_basis_0p4_poly_rls_0p25` | 1654.0441 +/- 1650.4341 | 414.1161 +/- 412.4571 |  |  | 320.0000 +/- 0.0000 | 5.9742 +/- 0.4099 |

`final_window_mse` best-D18-vs-best-MLP diff: -116.5097 +/- 115.5390; wins/losses/ties 0/3/0; best-D18 counts {'d18_core_0p25_basis_0p6_poly_rls_0p1': 3}.

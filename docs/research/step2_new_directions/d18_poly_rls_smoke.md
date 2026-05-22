# D18 Simple Universal Resource-Basis Results

Protocol: 1 paired seeds, 120 online steps, final window 40. Candidate configs: d18_core_0p5_basis_0p4_poly_rls_0p25.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2590 +/- 0.0000 | 0.2945 +/- 0.0000 |  |  |  | 1.4073 +/- 0.0000 |
| `mlp_h128` | 0.2785 +/- 0.0000 | 0.3061 +/- 0.0000 |  |  |  | 1.0942 +/- 0.0000 |
| `mlp_h64_64` | 0.2920 +/- 0.0000 | 0.3426 +/- 0.0000 |  |  |  | 1.0259 +/- 0.0000 |
| `d18_core_0p5_basis_0p4_poly_rls_0p25` | 0.2480 +/- 0.0000 | 0.2497 +/- 0.0000 |  |  | 24.0000 +/- 0.0000 | 0.2348 +/- 0.0000 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0110 +/- 0.0000; wins/losses/ties 1/0/0; best-D18 counts {'d18_core_0p5_basis_0p4_poly_rls_0p25': 1}.

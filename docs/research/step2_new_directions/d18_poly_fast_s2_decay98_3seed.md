# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p5_basis_0p4_poly_0p25, d18_core_0p5_basis_0p4_poly_0p4_decay, d18_core_0p5_basis_0p4_poly_0p6_decay.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.2758 +/- 0.5934 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.1668 +/- 0.6605 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.7886 +/- 0.2320 |
| `d18_core_0p5_basis_0p4_poly_0p25` | 1.2109 +/- 0.4351 | 1.1730 +/- 0.2971 |  |  | 320.0000 +/- 0.0000 | 3.6013 +/- 0.4671 |
| `d18_core_0p5_basis_0p4_poly_0p4_decay` | 1.1379 +/- 0.5921 | 1.1917 +/- 0.2771 |  |  | 320.0000 +/- 0.0000 | 4.4167 +/- 0.5298 |
| `d18_core_0p5_basis_0p4_poly_0p6_decay` | 1.3100 +/- 0.6349 | 1.3415 +/- 0.2880 |  |  | 320.0000 +/- 0.0000 | 4.4926 +/- 0.6036 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.1053 +/- 0.1375; wins/losses/ties 1/2/0; best-D18 counts {'d18_core_0p5_basis_0p4_poly_0p25': 1, 'd18_core_0p5_basis_0p4_poly_0p4_decay': 2}.

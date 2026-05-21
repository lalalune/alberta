# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p5_basis_0p4_poly_0p25, d18_core_0p5_basis_0p4_poly_0p4_decay, d18_core_0p5_basis_0p4_poly_0p6_decay.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.2025 +/- 0.6491 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.3675 +/- 0.6725 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.8639 +/- 0.2800 |
| `d18_core_0p5_basis_0p4_poly_0p25` | 1.0482 +/- 0.4840 | 1.0955 +/- 0.2789 |  |  | 320.0000 +/- 0.0000 | 3.4453 +/- 0.2826 |
| `d18_core_0p5_basis_0p4_poly_0p4_decay` | 1.0273 +/- 0.5527 | 1.1033 +/- 0.2717 |  |  | 320.0000 +/- 0.0000 | 4.1441 +/- 0.4553 |
| `d18_core_0p5_basis_0p4_poly_0p6_decay` | 1.0603 +/- 0.5512 | 1.1283 +/- 0.2707 |  |  | 320.0000 +/- 0.0000 | 3.7386 +/- 0.4519 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0404 +/- 0.1315; wins/losses/ties 1/2/0; best-D18 counts {'d18_core_0p5_basis_0p4_poly_0p25': 1, 'd18_core_0p5_basis_0p4_poly_0p4_decay': 2}.

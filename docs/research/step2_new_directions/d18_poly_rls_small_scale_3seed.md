# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p5_basis_0p4, d18_core_0p5_basis_0p4_poly_rls_0p05, d18_core_0p5_basis_0p4_poly_rls_0p1, d18_core_0p5_basis_0p4_poly_rls_0p15, d18_core_0p5_basis_0p4_poly_rls_0p25.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 0.5148 +/- 0.0371 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 0.7026 +/- 0.1821 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 0.5374 +/- 0.0125 |
| `d18_core_0p5_basis_0p4` | 0.0320 +/- 0.0006 | 0.0457 +/- 0.0006 | 0.8978 +/- 0.0022 | 0.9592 +/- 0.0039 | 320.0000 +/- 0.0000 | 2.7150 +/- 0.4166 |
| `d18_core_0p5_basis_0p4_poly_rls_0p05` | 0.0347 +/- 0.0004 | 0.0496 +/- 0.0011 | 0.8978 +/- 0.0106 | 0.9431 +/- 0.0051 | 320.0000 +/- 0.0000 | 2.9850 +/- 0.2709 |
| `d18_core_0p5_basis_0p4_poly_rls_0p1` | 0.0400 +/- 0.0007 | 0.0587 +/- 0.0031 | 0.8878 +/- 0.0080 | 0.9295 +/- 0.0075 | 320.0000 +/- 0.0000 | 2.5836 +/- 0.1824 |
| `d18_core_0p5_basis_0p4_poly_rls_0p15` | 0.0461 +/- 0.0009 | 0.0710 +/- 0.0061 | 0.8767 +/- 0.0088 | 0.9215 +/- 0.0055 | 320.0000 +/- 0.0000 | 3.0645 +/- 0.3573 |
| `d18_core_0p5_basis_0p4_poly_rls_0p25` | 0.0604 +/- 0.0026 | 0.0984 +/- 0.0101 | 0.8533 +/- 0.0145 | 0.9103 +/- 0.0087 | 320.0000 +/- 0.0000 | 2.5589 +/- 0.3352 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0061 +/- 0.0011; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0519 +/- 0.0054; wins/losses/ties 3/0/0.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.7949 +/- 0.4844 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.0015 +/- 0.5952 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.6887 +/- 0.1947 |
| `d18_core_0p5_basis_0p4` | 1.0385 +/- 0.5953 | 1.1072 +/- 0.2793 |  |  | 320.0000 +/- 0.0000 | 2.3681 +/- 0.2338 |
| `d18_core_0p5_basis_0p4_poly_rls_0p05` | 0.9962 +/- 0.5269 | 1.0244 +/- 0.2555 |  |  | 320.0000 +/- 0.0000 | 2.8242 +/- 0.2662 |
| `d18_core_0p5_basis_0p4_poly_rls_0p1` | 0.9871 +/- 0.4668 | 0.9763 +/- 0.2415 |  |  | 320.0000 +/- 0.0000 | 2.1650 +/- 0.1061 |
| `d18_core_0p5_basis_0p4_poly_rls_0p15` | 0.9821 +/- 0.4187 | 0.9435 +/- 0.2332 |  |  | 320.0000 +/- 0.0000 | 2.3580 +/- 0.1688 |
| `d18_core_0p5_basis_0p4_poly_rls_0p25` | 0.9521 +/- 0.3403 | 0.8922 +/- 0.2228 |  |  | 320.0000 +/- 0.0000 | 2.4282 +/- 0.3390 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.1081 +/- 0.2483; wins/losses/ties 1/2/0; best-D18 counts {'d18_core_0p5_basis_0p4': 2, 'd18_core_0p5_basis_0p4_poly_rls_0p25': 1}.

# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p5_basis_0p4, d18_core_0p5_basis_0p4_poly_rls_0p25, d18_core_0p5_basis_0p4_poly_rls_0p4, d18_core_0p5_basis_0p4_poly_rls_0p6.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 0.6024 +/- 0.0697 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 1.0631 +/- 0.2459 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 0.6844 +/- 0.0557 |
| `d18_core_0p5_basis_0p4` | 0.0320 +/- 0.0006 | 0.0457 +/- 0.0006 | 0.8978 +/- 0.0022 | 0.9592 +/- 0.0039 | 320.0000 +/- 0.0000 | 3.3425 +/- 0.3155 |
| `d18_core_0p5_basis_0p4_poly_rls_0p25` | 0.0598 +/- 0.0026 | 0.0973 +/- 0.0100 | 0.8611 +/- 0.0144 | 0.9103 +/- 0.0111 | 320.0000 +/- 0.0000 | 3.4931 +/- 0.2665 |
| `d18_core_0p5_basis_0p4_poly_rls_0p4` | 0.0857 +/- 0.0095 | 0.1484 +/- 0.0192 | 0.8322 +/- 0.0179 | 0.8961 +/- 0.0126 | 320.0000 +/- 0.0000 | 4.3442 +/- 0.8924 |
| `d18_core_0p5_basis_0p4_poly_rls_0p6` | 0.1419 +/- 0.0362 | 0.2458 +/- 0.0495 | 0.8122 +/- 0.0216 | 0.8794 +/- 0.0147 | 320.0000 +/- 0.0000 | 4.2553 +/- 0.3499 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0061 +/- 0.0011; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0519 +/- 0.0054; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.5055 +/- 0.2194 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.4880 +/- 0.1881 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.4443 +/- 0.0960 |
| `d18_core_0p5_basis_0p4` | 0.2411 +/- 0.0642 | 0.1554 +/- 0.0233 |  |  | 320.0000 +/- 0.0000 | 1.9773 +/- 0.5098 |
| `d18_core_0p5_basis_0p4_poly_rls_0p25` | 0.2320 +/- 0.0531 | 0.1570 +/- 0.0213 |  |  | 320.0000 +/- 0.0000 | 4.5996 +/- 1.4093 |
| `d18_core_0p5_basis_0p4_poly_rls_0p4` | 0.2305 +/- 0.0557 | 0.1584 +/- 0.0217 |  |  | 320.0000 +/- 0.0000 | 3.7794 +/- 1.2462 |
| `d18_core_0p5_basis_0p4_poly_rls_0p6` | 0.2321 +/- 0.0557 | 0.1647 +/- 0.0219 |  |  | 320.0000 +/- 0.0000 | 3.1584 +/- 0.1373 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0445 +/- 0.0400; wins/losses/ties 2/1/0; best-D18 counts {'d18_core_0p5_basis_0p4': 1, 'd18_core_0p5_basis_0p4_poly_rls_0p25': 1, 'd18_core_0p5_basis_0p4_poly_rls_0p4': 1}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.0127 +/- 0.7215 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.8415 +/- 0.4765 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.4828 +/- 0.1645 |
| `d18_core_0p5_basis_0p4` | 1.0385 +/- 0.5953 | 1.1072 +/- 0.2793 |  |  | 320.0000 +/- 0.0000 | 2.6874 +/- 0.4352 |
| `d18_core_0p5_basis_0p4_poly_rls_0p25` | 0.9544 +/- 0.3396 | 0.8932 +/- 0.2237 |  |  | 320.0000 +/- 0.0000 | 3.4999 +/- 0.9495 |
| `d18_core_0p5_basis_0p4_poly_rls_0p4` | 0.8885 +/- 0.2642 | 0.8316 +/- 0.2115 |  |  | 320.0000 +/- 0.0000 | 4.6537 +/- 1.5537 |
| `d18_core_0p5_basis_0p4_poly_rls_0p6` | 0.8073 +/- 0.2052 | 0.7667 +/- 0.1942 |  |  | 320.0000 +/- 0.0000 | 3.3353 +/- 0.4935 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.2462 +/- 0.3864; wins/losses/ties 1/2/0; best-D18 counts {'d18_core_0p5_basis_0p4': 2, 'd18_core_0p5_basis_0p4_poly_rls_0p6': 1}.

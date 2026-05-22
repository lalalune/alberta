# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p25_basis_0p6, d18_core_0p25_basis_0p6_poly_rls_0p05, d18_core_0p25_basis_0p6_poly_rls_0p1, d18_core_0p25_basis_0p6_poly_rls_0p15.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.0181 +/- 0.1185 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 1.6876 +/- 0.4184 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 1.5978 +/- 0.3594 |
| `d18_core_0p25_basis_0p6` | 0.0378 +/- 0.0007 | 0.0503 +/- 0.0006 | 0.8833 +/- 0.0077 | 0.9450 +/- 0.0062 | 320.0000 +/- 0.0000 | 6.1703 +/- 1.3689 |
| `d18_core_0p25_basis_0p6_poly_rls_0p05` | 0.0403 +/- 0.0004 | 0.0545 +/- 0.0012 | 0.8767 +/- 0.0077 | 0.9301 +/- 0.0080 | 320.0000 +/- 0.0000 | 6.7830 +/- 1.3070 |
| `d18_core_0p25_basis_0p6_poly_rls_0p1` | 0.0453 +/- 0.0006 | 0.0641 +/- 0.0033 | 0.8644 +/- 0.0062 | 0.9215 +/- 0.0048 | 320.0000 +/- 0.0000 | 4.4507 +/- 0.6548 |
| `d18_core_0p25_basis_0p6_poly_rls_0p15` | 0.0510 +/- 0.0009 | 0.0772 +/- 0.0068 | 0.8544 +/- 0.0087 | 0.9134 +/- 0.0097 | 320.0000 +/- 0.0000 | 5.4498 +/- 0.6530 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0003 +/- 0.0013; wins/losses/ties 2/1/0; best-D18 counts {'d18_core_0p25_basis_0p6': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0377 +/- 0.0073; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.7058 +/- 0.1846 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.7917 +/- 0.2856 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.6153 +/- 0.0449 |
| `d18_core_0p25_basis_0p6` | 0.2277 +/- 0.0676 | 0.1478 +/- 0.0237 |  |  | 320.0000 +/- 0.0000 | 4.0488 +/- 1.0849 |
| `d18_core_0p25_basis_0p6_poly_rls_0p05` | 0.2257 +/- 0.0611 | 0.1479 +/- 0.0222 |  |  | 320.0000 +/- 0.0000 | 6.0880 +/- 0.7200 |
| `d18_core_0p25_basis_0p6_poly_rls_0p1` | 0.2248 +/- 0.0638 | 0.1481 +/- 0.0230 |  |  | 320.0000 +/- 0.0000 | 4.5349 +/- 0.5332 |
| `d18_core_0p25_basis_0p6_poly_rls_0p15` | 0.2237 +/- 0.0618 | 0.1492 +/- 0.0234 |  |  | 320.0000 +/- 0.0000 | 5.7049 +/- 1.6158 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0501 +/- 0.0324; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p25_basis_0p6': 1, 'd18_core_0p25_basis_0p6_poly_rls_0p1': 1, 'd18_core_0p25_basis_0p6_poly_rls_0p15': 1}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.2067 +/- 0.9893 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.8073 +/- 0.5670 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.4526 +/- 0.1605 |
| `d18_core_0p25_basis_0p6` | 1.0380 +/- 0.6272 | 1.1072 +/- 0.2764 |  |  | 320.0000 +/- 0.0000 | 1.7909 +/- 0.4277 |
| `d18_core_0p25_basis_0p6_poly_rls_0p05` | 0.9921 +/- 0.5518 | 1.0229 +/- 0.2529 |  |  | 320.0000 +/- 0.0000 | 2.0517 +/- 0.1835 |
| `d18_core_0p25_basis_0p6_poly_rls_0p1` | 0.9794 +/- 0.4861 | 0.9725 +/- 0.2387 |  |  | 320.0000 +/- 0.0000 | 1.8417 +/- 0.1638 |
| `d18_core_0p25_basis_0p6_poly_rls_0p15` | 0.9715 +/- 0.4369 | 0.9389 +/- 0.2310 |  |  | 320.0000 +/- 0.0000 | 1.8394 +/- 0.1001 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0587 +/- 0.1666; wins/losses/ties 1/2/0; best-D18 counts {'d18_core_0p25_basis_0p6': 2, 'd18_core_0p25_basis_0p6_poly_rls_0p15': 1}.

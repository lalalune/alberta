# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p5_basis_0p4, d18_core_0p5_basis_0p4_poly_0p25, d18_core_0p5_basis_0p4_poly_0p4, d18_core_0p5_basis_0p4_poly_0p4_decay, d18_core_0p5_basis_0p4_poly_0p6_decay.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.2962 +/- 0.2399 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 1.1047 +/- 0.1923 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 0.7203 +/- 0.0658 |
| `d18_core_0p5_basis_0p4` | 0.0320 +/- 0.0006 | 0.0457 +/- 0.0006 | 0.8978 +/- 0.0022 | 0.9592 +/- 0.0039 | 320.0000 +/- 0.0000 | 4.2629 +/- 0.8336 |
| `d18_core_0p5_basis_0p4_poly_0p25` | 0.0364 +/- 0.0010 | 0.0506 +/- 0.0008 | 0.8922 +/- 0.0106 | 0.9468 +/- 0.0090 | 320.0000 +/- 0.0000 | 5.2006 +/- 2.4535 |
| `d18_core_0p5_basis_0p4_poly_0p4` | 0.0416 +/- 0.0016 | 0.0561 +/- 0.0012 | 0.8800 +/- 0.0120 | 0.9332 +/- 0.0132 | 320.0000 +/- 0.0000 | 4.2723 +/- 1.1275 |
| `d18_core_0p5_basis_0p4_poly_0p4_decay` | 0.0371 +/- 0.0005 | 0.0522 +/- 0.0008 | 0.8867 +/- 0.0096 | 0.9493 +/- 0.0077 | 320.0000 +/- 0.0000 | 4.3030 +/- 0.8304 |
| `d18_core_0p5_basis_0p4_poly_0p6_decay` | 0.0413 +/- 0.0007 | 0.0577 +/- 0.0011 | 0.8833 +/- 0.0100 | 0.9412 +/- 0.0071 | 320.0000 +/- 0.0000 | 4.5608 +/- 0.9734 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0061 +/- 0.0011; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0519 +/- 0.0054; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.7005 +/- 0.1596 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.5639 +/- 0.1836 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.4676 +/- 0.0573 |
| `d18_core_0p5_basis_0p4` | 0.2411 +/- 0.0642 | 0.1554 +/- 0.0233 |  |  | 320.0000 +/- 0.0000 | 3.4989 +/- 1.0006 |
| `d18_core_0p5_basis_0p4_poly_0p25` | 0.2702 +/- 0.0687 | 0.1695 +/- 0.0236 |  |  | 320.0000 +/- 0.0000 | 3.5391 +/- 0.3572 |
| `d18_core_0p5_basis_0p4_poly_0p4` | 0.3130 +/- 0.0870 | 0.1915 +/- 0.0277 |  |  | 320.0000 +/- 0.0000 | 5.6052 +/- 2.3849 |
| `d18_core_0p5_basis_0p4_poly_0p4_decay` | 0.2472 +/- 0.0756 | 0.1585 +/- 0.0256 |  |  | 320.0000 +/- 0.0000 | 4.4210 +/- 1.0443 |
| `d18_core_0p5_basis_0p4_poly_0p6_decay` | 0.2708 +/- 0.0888 | 0.1723 +/- 0.0280 |  |  | 320.0000 +/- 0.0000 | 4.5509 +/- 0.7878 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0361 +/- 0.0298; wins/losses/ties 2/1/0; best-D18 counts {'d18_core_0p5_basis_0p4': 2, 'd18_core_0p5_basis_0p4_poly_0p4_decay': 1}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.0674 +/- 0.6549 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.0691 +/- 0.5899 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 1.0591 +/- 0.4596 |
| `d18_core_0p5_basis_0p4` | 1.0385 +/- 0.5953 | 1.1072 +/- 0.2793 |  |  | 320.0000 +/- 0.0000 | 4.1135 +/- 0.9464 |
| `d18_core_0p5_basis_0p4_poly_0p25` | 1.0149 +/- 0.5306 | 1.0860 +/- 0.2749 |  |  | 320.0000 +/- 0.0000 | 4.2844 +/- 0.8410 |
| `d18_core_0p5_basis_0p4_poly_0p4` | 1.0335 +/- 0.5002 | 1.0899 +/- 0.2768 |  |  | 320.0000 +/- 0.0000 | 4.0000 +/- 0.3185 |
| `d18_core_0p5_basis_0p4_poly_0p4_decay` | 1.0033 +/- 0.5529 | 1.0832 +/- 0.2703 |  |  | 320.0000 +/- 0.0000 | 2.9619 +/- 0.5524 |
| `d18_core_0p5_basis_0p4_poly_0p6_decay` | 1.0016 +/- 0.5387 | 1.0814 +/- 0.2682 |  |  | 320.0000 +/- 0.0000 | 4.1451 +/- 0.2215 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0256 +/- 0.1147; wins/losses/ties 1/2/0; best-D18 counts {'d18_core_0p5_basis_0p4': 2, 'd18_core_0p5_basis_0p4_poly_0p4': 1}.

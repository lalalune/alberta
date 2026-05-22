# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_simple, d18_basis_half, d18_basis_quarter, d18_basis_third, d18_basis_0p4, d18_core_0p75_basis_0p4, d18_core_0p5_basis_0p4, d18_core_0p5_basis_0p4_poly_0p25, d18_core_0p5_basis_0p4_poly_0p4, d18_core_0p5_basis_0p4_poly_0p4_decay, d18_core_0p5_basis_0p4_poly_0p6_decay.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.5813 +/- 1.1323 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.0410 +/- 0.6627 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 1.0105 +/- 0.4373 |
| `d18_simple` | 1.7734 +/- 0.6953 | 1.8135 +/- 0.4606 |  |  | 320.0000 +/- 0.0000 | 2.6005 +/- 0.4381 |
| `d18_basis_half` | 1.2954 +/- 0.6191 | 1.3729 +/- 0.3462 |  |  | 320.0000 +/- 0.0000 | 3.4015 +/- 0.7757 |
| `d18_basis_quarter` | 1.1706 +/- 0.5982 | 1.2433 +/- 0.3117 |  |  | 320.0000 +/- 0.0000 | 2.5222 +/- 0.5670 |
| `d18_basis_third` | 1.2064 +/- 0.6066 | 1.2801 +/- 0.3207 |  |  | 320.0000 +/- 0.0000 | 3.6521 +/- 1.0924 |
| `d18_basis_0p4` | 1.2381 +/- 0.6096 | 1.3131 +/- 0.3295 |  |  | 320.0000 +/- 0.0000 | 2.6636 +/- 0.6317 |
| `d18_core_0p75_basis_0p4` | 1.1175 +/- 0.5926 | 1.1943 +/- 0.3019 |  |  | 320.0000 +/- 0.0000 | 2.7194 +/- 0.1744 |
| `d18_core_0p5_basis_0p4` | 1.0360 +/- 0.5939 | 1.1069 +/- 0.2787 |  |  | 320.0000 +/- 0.0000 | 1.8767 +/- 0.3021 |
| `d18_core_0p5_basis_0p4_poly_0p25` | 1.0158 +/- 0.5323 | 1.0861 +/- 0.2752 |  |  | 320.0000 +/- 0.0000 | 3.0910 +/- 0.1832 |
| `d18_core_0p5_basis_0p4_poly_0p4` | 1.0295 +/- 0.5021 | 1.0895 +/- 0.2768 |  |  | 320.0000 +/- 0.0000 | 2.4847 +/- 0.6575 |
| `d18_core_0p5_basis_0p4_poly_0p4_decay` | 1.0035 +/- 0.5506 | 1.0842 +/- 0.2717 |  |  | 320.0000 +/- 0.0000 | 5.3864 +/- 1.8976 |
| `d18_core_0p5_basis_0p4_poly_0p6_decay` | 1.0011 +/- 0.5373 | 1.0827 +/- 0.2691 |  |  | 320.0000 +/- 0.0000 | 2.9271 +/- 0.5791 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0249 +/- 0.1143; wins/losses/ties 1/2/0; best-D18 counts {'d18_core_0p5_basis_0p4': 2, 'd18_core_0p5_basis_0p4_poly_0p4': 1}.

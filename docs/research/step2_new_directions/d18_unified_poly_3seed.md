# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p25_basis_0p6, d18_core_0p25_basis_0p6_unified_0p05, d18_core_0p25_basis_0p6_unified_0p1, d18_core_0p25_basis_0p6_unified_0p2.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 2.1022 +/- 1.0998 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 2.4795 +/- 1.0610 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 2.0461 +/- 0.5798 |
| `d18_core_0p25_basis_0p6` | 1.0380 +/- 0.6272 | 1.1072 +/- 0.2764 |  |  | 320.0000 +/- 0.0000 | 12.2041 +/- 3.8036 |
| `d18_core_0p25_basis_0p6_unified_0p05` | 1.0344 +/- 0.6217 | 1.1061 +/- 0.2759 |  |  | 320.0000 +/- 0.0000 | 15.8697 +/- 3.3182 |
| `d18_core_0p25_basis_0p6_unified_0p1` | 1.0358 +/- 0.6163 | 1.1058 +/- 0.2743 |  |  | 320.0000 +/- 0.0000 | 11.4483 +/- 0.5286 |
| `d18_core_0p25_basis_0p6_unified_0p2` | 1.0361 +/- 0.6089 | 1.1064 +/- 0.2745 |  |  | 320.0000 +/- 0.0000 | 15.2447 +/- 3.3745 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0777 +/- 0.0305; wins/losses/ties 0/3/0; best-D18 counts {'d18_core_0p25_basis_0p6': 2, 'd18_core_0p25_basis_0p6_unified_0p2': 1}.

# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_simple, d18_basis_half, d18_basis_quarter.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier basis. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.2336 +/- 0.5159 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 2.4050 +/- 1.2742 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 1.2058 +/- 0.1331 |
| `d18_simple` | 0.0659 +/- 0.0019 | 0.0709 +/- 0.0012 | 0.8400 +/- 0.0208 | 0.8590 +/- 0.0057 | 320.0000 +/- 0.0000 | 4.3373 +/- 1.1943 |
| `d18_basis_half` | 0.0396 +/- 0.0013 | 0.0505 +/- 0.0007 | 0.9033 +/- 0.0069 | 0.9258 +/- 0.0130 | 320.0000 +/- 0.0000 | 6.2872 +/- 1.0661 |
| `d18_basis_quarter` | 0.0334 +/- 0.0010 | 0.0454 +/- 0.0007 | 0.9111 +/- 0.0022 | 0.9456 +/- 0.0077 | 320.0000 +/- 0.0000 | 4.0421 +/- 0.4266 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0047 +/- 0.0014; wins/losses/ties 3/0/0; best-D18 counts {'d18_basis_quarter': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0383 +/- 0.0089; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.9242 +/- 1.0781 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 3.0325 +/- 2.1867 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 2.0881 +/- 0.7033 |
| `d18_simple` | 0.3228 +/- 0.0770 | 0.2044 +/- 0.0303 |  |  | 320.0000 +/- 0.0000 | 8.3782 +/- 0.7245 |
| `d18_basis_half` | 0.2619 +/- 0.0530 | 0.1695 +/- 0.0222 |  |  | 320.0000 +/- 0.0000 | 7.7470 +/- 2.1770 |
| `d18_basis_quarter` | 0.2754 +/- 0.0593 | 0.1736 +/- 0.0222 |  |  | 320.0000 +/- 0.0000 | 7.1675 +/- 2.9629 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0098 +/- 0.0439; wins/losses/ties 1/2/0; best-D18 counts {'d18_basis_half': 3}.

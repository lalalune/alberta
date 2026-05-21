# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_basis_third, d18_basis_0p4.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier basis. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.9668 +/- 0.6332 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 2.9259 +/- 0.4815 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 3.8263 +/- 0.6206 |
| `d18_basis_third` | 0.0352 +/- 0.0011 | 0.0468 +/- 0.0007 | 0.9078 +/- 0.0022 | 0.9419 +/- 0.0087 | 320.0000 +/- 0.0000 | 14.3164 +/- 2.0693 |
| `d18_basis_0p4` | 0.0368 +/- 0.0012 | 0.0481 +/- 0.0007 | 0.9100 +/- 0.0051 | 0.9363 +/- 0.0120 | 320.0000 +/- 0.0000 | 16.3662 +/- 3.1337 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0029 +/- 0.0014; wins/losses/ties 3/0/0; best-D18 counts {'d18_basis_third': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0346 +/- 0.0096; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.5401 +/- 0.7804 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.2641 +/- 0.6469 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.1493 +/- 0.3021 |
| `d18_basis_third` | 0.2697 +/- 0.0610 | 0.1702 +/- 0.0229 |  |  | 320.0000 +/- 0.0000 | 4.3701 +/- 1.5135 |
| `d18_basis_0p4` | 0.2612 +/- 0.0521 | 0.1689 +/- 0.0217 |  |  | 320.0000 +/- 0.0000 | 5.4155 +/- 1.3840 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0117 +/- 0.0442; wins/losses/ties 1/2/0; best-D18 counts {'d18_basis_0p4': 2, 'd18_basis_third': 1}.

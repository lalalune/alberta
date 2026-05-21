# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_core_0p75_basis_0p4, d18_core_0p5_basis_0p4.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier basis. There is no output router and no MLP expert.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.2420 +/- 0.4013 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 1.3190 +/- 0.7167 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 0.9736 +/- 0.2190 |
| `d18_core_0p75_basis_0p4` | 0.0325 +/- 0.0009 | 0.0456 +/- 0.0007 | 0.9111 +/- 0.0059 | 0.9524 +/- 0.0062 | 320.0000 +/- 0.0000 | 5.7366 +/- 2.1959 |
| `d18_core_0p5_basis_0p4` | 0.0320 +/- 0.0006 | 0.0457 +/- 0.0006 | 0.9022 +/- 0.0048 | 0.9561 +/- 0.0059 | 320.0000 +/- 0.0000 | 5.3430 +/- 1.0963 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0062 +/- 0.0012; wins/losses/ties 3/0/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0489 +/- 0.0063; wins/losses/ties 3/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.2514 +/- 0.6843 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.4437 +/- 0.9791 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.0333 +/- 0.3731 |
| `d18_core_0p75_basis_0p4` | 0.2482 +/- 0.0609 | 0.1589 +/- 0.0226 |  |  | 320.0000 +/- 0.0000 | 2.6737 +/- 0.5425 |
| `d18_core_0p5_basis_0p4` | 0.2410 +/- 0.0600 | 0.1556 +/- 0.0219 |  |  | 320.0000 +/- 0.0000 | 3.6740 +/- 0.4735 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0307 +/- 0.0355; wins/losses/ties 2/1/0; best-D18 counts {'d18_core_0p5_basis_0p4': 3}.

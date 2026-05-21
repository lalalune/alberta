# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_basis_0p5, d18_step2_basis_0p6, d18_step2_core_0p4_basis_0p5, d18_step2_gain_l2_0p1.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.8786 +/- 0.1599 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.8605 +/- 0.1306 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.8299 +/- 0.0648 |
| `d18_step2_basis_0p5` | 0.0182 +/- 0.0006 | 0.0178 +/- 0.0003 | 0.9927 +/- 0.0004 | 0.1006 +/- 0.0004 | 320.0000 +/- 0.0000 | 4.7010 +/- 0.6606 |
| `d18_step2_basis_0p6` | 0.0180 +/- 0.0006 | 0.0176 +/- 0.0003 | 0.9923 +/- 0.0005 | 0.1006 +/- 0.0004 | 320.0000 +/- 0.0000 | 4.8315 +/- 0.6770 |
| `d18_step2_core_0p4_basis_0p5` | 0.0176 +/- 0.0007 | 0.0175 +/- 0.0003 | 0.9927 +/- 0.0004 | 0.1006 +/- 0.0004 | 320.0000 +/- 0.0000 | 5.2406 +/- 0.8052 |
| `d18_step2_gain_l2_0p1` | 0.0202 +/- 0.0006 | 0.0193 +/- 0.0003 | 0.9927 +/- 0.0004 | 0.1006 +/- 0.0004 | 320.0000 +/- 0.0000 | 4.9715 +/- 0.5917 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0146 +/- 0.0006; wins/losses/ties 0/10/0; best-D18 counts {'d18_step2_basis_0p6': 2, 'd18_step2_core_0p4_basis_0p5': 8}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0353 +/- 0.0119; wins/losses/ties 1/9/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.6847 +/- 0.0633 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.6815 +/- 0.0490 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.8247 +/- 0.1133 |
| `d18_step2_basis_0p5` | 0.0429 +/- 0.0015 | 0.0526 +/- 0.0009 | 0.8023 +/- 0.0140 | 0.8425 +/- 0.0124 | 320.0000 +/- 0.0000 | 5.0846 +/- 0.7354 |
| `d18_step2_basis_0p6` | 0.0440 +/- 0.0015 | 0.0538 +/- 0.0009 | 0.8017 +/- 0.0153 | 0.8429 +/- 0.0118 | 320.0000 +/- 0.0000 | 4.5657 +/- 0.8116 |
| `d18_step2_core_0p4_basis_0p5` | 0.0430 +/- 0.0014 | 0.0528 +/- 0.0008 | 0.7980 +/- 0.0144 | 0.8404 +/- 0.0130 | 320.0000 +/- 0.0000 | 5.2139 +/- 0.5822 |
| `d18_step2_gain_l2_0p1` | 0.0429 +/- 0.0015 | 0.0522 +/- 0.0009 | 0.8067 +/- 0.0135 | 0.8477 +/- 0.0125 | 320.0000 +/- 0.0000 | 4.7480 +/- 0.7837 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0053 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p5': 4, 'd18_step2_core_0p4_basis_0p5': 2, 'd18_step2_gain_l2_0p1': 4}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0132 +/- 0.0099; wins/losses/ties 6/4/0.

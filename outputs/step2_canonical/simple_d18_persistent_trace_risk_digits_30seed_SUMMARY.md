# D18 Simple Universal Resource-Basis Results

Protocol: 30 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_gain_l2_0p1.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0001 | 0.0090 +/- 0.0000 | 0.9861 +/- 0.0007 | 0.1137 +/- 0.0048 |  | 0.4947 +/- 0.0500 |
| `mlp_h128` | 0.0073 +/- 0.0001 | 0.0116 +/- 0.0001 | 0.9852 +/- 0.0007 | 0.1179 +/- 0.0052 |  | 0.4699 +/- 0.0386 |
| `mlp_h64_64` | 0.0030 +/- 0.0000 | 0.0063 +/- 0.0001 | 0.9910 +/- 0.0004 | 0.1002 +/- 0.0003 |  | 0.5160 +/- 0.0198 |
| `d18_step2_gain_l2_0p1` | 0.0015 +/- 0.0001 | 0.0019 +/- 0.0000 | 0.9926 +/- 0.0003 | 0.1002 +/- 0.0003 | 320.0000 +/- 0.0000 | 2.4959 +/- 0.0711 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0015 +/- 0.0001; wins/losses/ties 30/0/0; best-D18 counts {'d18_step2_gain_l2_0p1': 30}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0211 +/- 0.0057; wins/losses/ties 6/24/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0496 +/- 0.0008 | 0.0604 +/- 0.0004 | 0.7804 +/- 0.0080 | 0.8163 +/- 0.0075 |  | 0.3578 +/- 0.0157 |
| `mlp_h128` | 0.0489 +/- 0.0008 | 0.0597 +/- 0.0004 | 0.7981 +/- 0.0075 | 0.8345 +/- 0.0073 |  | 0.3883 +/- 0.0188 |
| `mlp_h64_64` | 0.0527 +/- 0.0009 | 0.0647 +/- 0.0005 | 0.7460 +/- 0.0078 | 0.7957 +/- 0.0084 |  | 0.4022 +/- 0.0140 |
| `d18_step2_gain_l2_0p1` | 0.0384 +/- 0.0015 | 0.0552 +/- 0.0009 | 0.8078 +/- 0.0075 | 0.8432 +/- 0.0078 | 320.0000 +/- 0.0000 | 2.1707 +/- 0.0581 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0100 +/- 0.0010; wins/losses/ties 29/1/0; best-D18 counts {'d18_step2_gain_l2_0p1': 30}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0062 +/- 0.0048; wins/losses/ties 16/14/0.

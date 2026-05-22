# D18 Simple Universal Resource-Basis Results

Protocol: 30 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0001 | 0.0090 +/- 0.0000 | 0.9861 +/- 0.0007 | 0.1137 +/- 0.0048 |  | 0.3352 +/- 0.0233 |
| `mlp_h128` | 0.0073 +/- 0.0001 | 0.0116 +/- 0.0001 | 0.9852 +/- 0.0007 | 0.1179 +/- 0.0052 |  | 0.3626 +/- 0.0229 |
| `mlp_h64_64` | 0.0030 +/- 0.0000 | 0.0063 +/- 0.0001 | 0.9910 +/- 0.0004 | 0.1002 +/- 0.0003 |  | 0.3611 +/- 0.0103 |
| `d18_step2_canonical` | 0.0017 +/- 0.0001 | 0.0021 +/- 0.0000 | 0.9914 +/- 0.0004 | 0.8588 +/- 0.0032 | 320.0000 +/- 0.0000 | 1.9698 +/- 0.0311 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0013 +/- 0.0001; wins/losses/ties 30/0/0; best-D18 counts {'d18_step2_canonical': 30}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.7374 +/- 0.0069; wins/losses/ties 30/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0496 +/- 0.0008 | 0.0604 +/- 0.0004 | 0.7804 +/- 0.0080 | 0.8163 +/- 0.0075 |  | 0.3286 +/- 0.0077 |
| `mlp_h128` | 0.0489 +/- 0.0008 | 0.0597 +/- 0.0004 | 0.7981 +/- 0.0075 | 0.8345 +/- 0.0073 |  | 0.3423 +/- 0.0088 |
| `mlp_h64_64` | 0.0527 +/- 0.0009 | 0.0647 +/- 0.0005 | 0.7460 +/- 0.0078 | 0.7957 +/- 0.0084 |  | 0.3831 +/- 0.0153 |
| `d18_step2_canonical` | 0.0384 +/- 0.0015 | 0.0559 +/- 0.0009 | 0.8082 +/- 0.0075 | 0.8461 +/- 0.0065 | 320.0000 +/- 0.0000 | 2.0278 +/- 0.0227 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0101 +/- 0.0010; wins/losses/ties 29/1/0; best-D18 counts {'d18_step2_canonical': 30}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0092 +/- 0.0047; wins/losses/ties 20/10/0.

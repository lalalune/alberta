# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.4525 +/- 0.0152 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.5133 +/- 0.0419 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.5054 +/- 0.0309 |
| `d18_step2_canonical` | 0.0429 +/- 0.0015 | 0.0523 +/- 0.0009 | 0.7920 +/- 0.0146 | 0.8328 +/- 0.0136 | 320.0000 +/- 0.0000 | 2.4582 +/- 0.0710 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0051 +/- 0.0009; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0032 +/- 0.0102; wins/losses/ties 5/5/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.5180 +/- 0.2857 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.4292 +/- 0.1780 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.3927 +/- 0.0871 |
| `d18_step2_canonical` | 0.2073 +/- 0.0283 | 0.1684 +/- 0.0130 |  |  | 320.0000 +/- 0.0000 | 1.9443 +/- 0.0680 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0620 +/- 0.0211; wins/losses/ties 9/1/0; best-D18 counts {'d18_step2_canonical': 10}.

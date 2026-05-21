# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_simplex_trace.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.5151 +/- 0.1335 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.5592 +/- 0.1308 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.4660 +/- 0.0396 |
| `d18_step2_simplex_trace` | 0.0014 +/- 0.0001 | 0.0018 +/- 0.0000 | 0.9930 +/- 0.0003 | 0.1341 +/- 0.0121 | 320.0000 +/- 0.0000 | 1.9976 +/- 0.0459 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0015 +/- 0.0001; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex_trace': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0017 +/- 0.0038; wins/losses/ties 5/5/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.3688 +/- 0.0198 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.4077 +/- 0.0232 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.4546 +/- 0.0222 |
| `d18_step2_simplex_trace` | 0.0469 +/- 0.0035 | 0.0699 +/- 0.0021 | 0.7653 +/- 0.0174 | 0.8195 +/- 0.0185 | 320.0000 +/- 0.0000 | 1.9737 +/- 0.0314 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0011 +/- 0.0028; wins/losses/ties 5/5/0; best-D18 counts {'d18_step2_simplex_trace': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0165 +/- 0.0128; wins/losses/ties 4/6/0.

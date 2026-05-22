# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_context_trace.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.3910 +/- 0.0636 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.4002 +/- 0.0610 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.3685 +/- 0.0144 |
| `d18_step2_context_trace` | 0.0019 +/- 0.0001 | 0.0024 +/- 0.0001 | 0.9907 +/- 0.0007 | 0.1006 +/- 0.0004 | 320.0000 +/- 0.0000 | 1.9382 +/- 0.0662 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0010 +/- 0.0002; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_context_trace': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0353 +/- 0.0119; wins/losses/ties 1/9/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.5859 +/- 0.0670 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.5273 +/- 0.0419 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.5353 +/- 0.0301 |
| `d18_step2_context_trace` | 0.0398 +/- 0.0029 | 0.0555 +/- 0.0019 | 0.8010 +/- 0.0145 | 0.8436 +/- 0.0124 | 320.0000 +/- 0.0000 | 3.7104 +/- 0.2453 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0082 +/- 0.0022; wins/losses/ties 9/1/0; best-D18 counts {'d18_step2_context_trace': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0076 +/- 0.0095; wins/losses/ties 6/4/0.

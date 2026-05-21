# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_simplex.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.7028 +/- 0.1159 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.6635 +/- 0.1728 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.4936 +/- 0.0550 |
| `d18_step2_simplex` | 0.0015 +/- 0.0001 | 0.0021 +/- 0.0001 | 0.9927 +/- 0.0004 | 0.1006 +/- 0.0004 | 320.0000 +/- 0.0000 | 2.3732 +/- 0.1344 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0014 +/- 0.0001; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_simplex': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0353 +/- 0.0119; wins/losses/ties 1/9/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.3934 +/- 0.0189 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.3514 +/- 0.0154 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.3757 +/- 0.0109 |
| `d18_step2_simplex` | 0.0399 +/- 0.0029 | 0.0557 +/- 0.0019 | 0.8007 +/- 0.0143 | 0.8434 +/- 0.0126 | 320.0000 +/- 0.0000 | 2.0095 +/- 0.0228 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0081 +/- 0.0021; wins/losses/ties 9/1/0; best-D18 counts {'d18_step2_simplex': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0074 +/- 0.0095; wins/losses/ties 6/4/0.

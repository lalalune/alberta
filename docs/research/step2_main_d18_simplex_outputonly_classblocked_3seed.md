# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical, d18_step2_simplex, d18_step2_simplex_trace.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 1.6877 +/- 1.0712 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 2.0475 +/- 1.4821 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 0.8037 +/- 0.1940 |
| `d18_step2_canonical` | 0.0045 +/- 0.0000 | 0.0040 +/- 0.0001 | 0.9889 +/- 0.0011 | 0.2189 +/- 0.0242 | 320.0000 +/- 0.0000 | 4.6227 +/- 1.1179 |
| `d18_step2_simplex` | 0.0020 +/- 0.0000 | 0.0031 +/- 0.0002 | 0.9900 +/- 0.0000 | 0.2171 +/- 0.0198 | 320.0000 +/- 0.0000 | 4.4841 +/- 0.9739 |
| `d18_step2_simplex_trace` | 0.0013 +/- 0.0000 | 0.0019 +/- 0.0001 | 0.9933 +/- 0.0000 | 0.1664 +/- 0.0284 | 320.0000 +/- 0.0000 | 4.8341 +/- 0.8272 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0017 +/- 0.0001; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_simplex_trace': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0557 +/- 0.0130; wins/losses/ties 3/0/0.

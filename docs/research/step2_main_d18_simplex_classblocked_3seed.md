# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical, d18_step2_simplex, d18_step2_simplex_trace.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 2.2128 +/- 1.0306 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 2.2755 +/- 1.2087 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 1.6265 +/- 0.3406 |
| `d18_step2_canonical` | 0.0045 +/- 0.0000 | 0.0040 +/- 0.0001 | 0.9889 +/- 0.0011 | 0.2189 +/- 0.0242 | 320.0000 +/- 0.0000 | 8.9127 +/- 1.3895 |
| `d18_step2_simplex` | 0.0049 +/- 0.0004 | 0.0044 +/- 0.0002 | 0.9756 +/- 0.0022 | 0.3599 +/- 0.0266 | 320.0000 +/- 0.0000 | 21.4666 +/- 7.7244 |
| `d18_step2_simplex_trace` | 0.0036 +/- 0.0006 | 0.0032 +/- 0.0001 | 0.9822 +/- 0.0029 | 0.3358 +/- 0.0251 | 320.0000 +/- 0.0000 | 9.2933 +/- 0.9692 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0003 +/- 0.0004; wins/losses/ties 1/2/0; best-D18 counts {'d18_step2_simplex': 1, 'd18_step2_simplex_trace': 2}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.2059 +/- 0.0436; wins/losses/ties 3/0/0.

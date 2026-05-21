# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical, d18_step2_fast_residual_trace.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 2.7685 +/- 0.9923 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 2.6328 +/- 0.7493 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 2.3869 +/- 0.0923 |
| `d18_step2_canonical` | 0.0045 +/- 0.0000 | 0.0040 +/- 0.0001 | 0.9889 +/- 0.0011 | 0.2189 +/- 0.0242 | 320.0000 +/- 0.0000 | 10.7839 +/- 2.7523 |
| `d18_step2_fast_residual_trace` | 0.0043 +/- 0.0001 | 0.0038 +/- 0.0001 | 0.9911 +/- 0.0011 | 0.2096 +/- 0.0215 | 320.0000 +/- 0.0000 | 11.1487 +/- 1.0732 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0013 +/- 0.0001; wins/losses/ties 0/3/0; best-D18 counts {'d18_step2_fast_residual_trace': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0550 +/- 0.0136; wins/losses/ties 3/0/0.

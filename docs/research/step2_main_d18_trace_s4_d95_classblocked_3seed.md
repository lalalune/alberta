# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 3.6546 +/- 2.5038 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 2.2834 +/- 1.2307 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 2.0227 +/- 0.2153 |
| `d18_step2_canonical` | 0.0038 +/- 0.0001 | 0.0033 +/- 0.0000 | 0.9933 +/- 0.0000 | 0.1973 +/- 0.0285 | 320.0000 +/- 0.0000 | 10.0262 +/- 2.1070 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0008 +/- 0.0001; wins/losses/ties 0/3/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0315 +/- 0.0095; wins/losses/ties 3/0/0.

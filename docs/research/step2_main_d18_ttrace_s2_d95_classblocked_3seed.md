# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 4.8251 +/- 0.1626 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 5.5249 +/- 0.7168 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 5.1381 +/- 0.2826 |
| `d18_step2_canonical` | 0.0062 +/- 0.0003 | 0.0057 +/- 0.0002 | 0.9900 +/- 0.0019 | 0.1002 +/- 0.0000 | 320.0000 +/- 0.0000 | 19.5164 +/- 6.7516 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0032 +/- 0.0004; wins/losses/ties 0/3/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0656 +/- 0.0208; wins/losses/ties 0/3/0.

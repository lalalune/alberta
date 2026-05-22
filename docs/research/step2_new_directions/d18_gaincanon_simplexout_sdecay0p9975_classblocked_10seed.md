# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.7502 +/- 0.1757 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.7616 +/- 0.1701 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.6298 +/- 0.0417 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0023 +/- 0.0003 | 0.0028 +/- 0.0001 | 0.9883 +/- 0.0014 | 0.1683 +/- 0.0147 | 320.0000 +/- 0.0000 | 2.9540 +/- 0.2094 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0005 +/- 0.0003; wins/losses/ties 8/2/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0325 +/- 0.0070; wins/losses/ties 10/0/0.

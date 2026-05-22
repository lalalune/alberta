# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.6575 +/- 0.1718 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.6110 +/- 0.1356 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.5132 +/- 0.0326 |
| `d18_step2_canonical` | 0.0047 +/- 0.0003 | 0.0041 +/- 0.0001 | 0.9767 +/- 0.0015 | 0.3174 +/- 0.0122 | 320.0000 +/- 0.0000 | 2.7345 +/- 0.3640 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0018 +/- 0.0002; wins/losses/ties 0/10/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.1816 +/- 0.0077; wins/losses/ties 10/0/0.

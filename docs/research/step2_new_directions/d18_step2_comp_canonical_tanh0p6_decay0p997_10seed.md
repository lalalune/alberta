# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 2.2778 +/- 1.7682 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 1.8871 +/- 1.3780 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 1.3940 +/- 0.7191 |
| `d18_step2_canonical` | 0.2044 +/- 0.0281 | 0.1671 +/- 0.0128 |  |  | 320.0000 +/- 0.0000 | 5.0843 +/- 1.1665 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0650 +/- 0.0218; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 1.8724 +/- 1.4088 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 1.5538 +/- 1.0646 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 1.2239 +/- 0.5749 |
| `d18_step2_canonical` | 0.2020 +/- 0.0281 | 0.1664 +/- 0.0126 |  |  | 320.0000 +/- 0.0000 | 4.1817 +/- 0.6533 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0673 +/- 0.0224; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

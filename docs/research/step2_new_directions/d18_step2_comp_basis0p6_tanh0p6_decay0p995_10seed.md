# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_basis_0p6.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.3491 +/- 0.1063 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.3370 +/- 0.0922 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.3154 +/- 0.0359 |
| `d18_step2_basis_0p6` | 0.2038 +/- 0.0299 | 0.1689 +/- 0.0129 |  |  | 320.0000 +/- 0.0000 | 1.4351 +/- 0.0256 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0655 +/- 0.0227; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p6': 10}.

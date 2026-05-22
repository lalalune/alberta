# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.3783 +/- 0.1427 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.4181 +/- 0.1501 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.4227 +/- 0.0824 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2362 +/- 0.0339 | 0.1877 +/- 0.0153 |  |  | 320.0000 +/- 0.0000 | 1.7238 +/- 0.1117 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0331 +/- 0.0189; wins/losses/ties 8/2/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.

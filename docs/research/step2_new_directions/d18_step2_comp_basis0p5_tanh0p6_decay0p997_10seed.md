# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_basis_0p5.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 2.2115 +/- 1.6286 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 1.3220 +/- 0.7691 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 1.2013 +/- 0.4376 |
| `d18_step2_basis_0p5` | 0.2037 +/- 0.0291 | 0.1674 +/- 0.0129 |  |  | 320.0000 +/- 0.0000 | 5.0424 +/- 0.8922 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0656 +/- 0.0218; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_basis_0p5': 10}.

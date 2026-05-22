# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical, d18_step2_basis_0p5, d18_step2_basis_0p6, d18_step2_core_0p4_basis_0p5, d18_step2_gain_l2_0p1, d18_step2_no_unified, d18_step2_no_poly.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.2487 +/- 0.0676 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.2308 +/- 0.0583 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.2359 +/- 0.0269 |
| `d18_step2_canonical` | 0.2244 +/- 0.0285 | 0.1764 +/- 0.0139 |  |  | 320.0000 +/- 0.0000 | 1.3697 +/- 0.0608 |
| `d18_step2_basis_0p5` | 0.2205 +/- 0.0281 | 0.1752 +/- 0.0135 |  |  | 320.0000 +/- 0.0000 | 1.4234 +/- 0.0893 |
| `d18_step2_basis_0p6` | 0.2181 +/- 0.0286 | 0.1741 +/- 0.0136 |  |  | 320.0000 +/- 0.0000 | 1.4067 +/- 0.0934 |
| `d18_step2_core_0p4_basis_0p5` | 0.2206 +/- 0.0281 | 0.1744 +/- 0.0136 |  |  | 320.0000 +/- 0.0000 | 1.4328 +/- 0.0849 |
| `d18_step2_gain_l2_0p1` | 0.2272 +/- 0.0280 | 0.1788 +/- 0.0140 |  |  | 320.0000 +/- 0.0000 | 1.4390 +/- 0.0811 |
| `d18_step2_no_unified` | 0.2323 +/- 0.0287 | 0.1819 +/- 0.0142 |  |  | 320.0000 +/- 0.0000 | 1.4084 +/- 0.1464 |
| `d18_step2_no_poly` | 0.2292 +/- 0.0307 | 0.1815 +/- 0.0144 |  |  | 320.0000 +/- 0.0000 | 1.4135 +/- 0.0880 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0537 +/- 0.0200; wins/losses/ties 9/1/0; best-D18 counts {'d18_step2_basis_0p5': 1, 'd18_step2_basis_0p6': 7, 'd18_step2_canonical': 1, 'd18_step2_gain_l2_0p1': 1}.

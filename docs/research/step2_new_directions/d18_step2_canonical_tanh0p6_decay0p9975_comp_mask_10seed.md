# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.3844 +/- 0.0147 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.4635 +/- 0.0355 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.4581 +/- 0.0212 |
| `d18_step2_canonical` | 0.0431 +/- 0.0015 | 0.0524 +/- 0.0009 | 0.7900 +/- 0.0146 | 0.8299 +/- 0.0139 | 320.0000 +/- 0.0000 | 2.1645 +/- 0.0778 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0049 +/- 0.0009; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0061 +/- 0.0106; wins/losses/ties 5/5/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.2870 +/- 0.1018 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.3393 +/- 0.1489 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.3751 +/- 0.1443 |
| `d18_step2_canonical` | 0.2057 +/- 0.0282 | 0.1676 +/- 0.0129 |  |  | 320.0000 +/- 0.0000 | 1.4981 +/- 0.0466 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0636 +/- 0.0214; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.

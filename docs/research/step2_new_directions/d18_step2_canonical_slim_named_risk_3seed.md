# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.2647 +/- 0.0024 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 0.3812 +/- 0.1043 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.3017 +/- 0.0092 |
| `d18_step2_canonical` | 0.0456 +/- 0.0013 | 0.0537 +/- 0.0013 | 0.7844 +/- 0.0238 | 0.8281 +/- 0.0204 | 320.0000 +/- 0.0000 | 1.8159 +/- 0.0312 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0022 +/- 0.0006; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0080 +/- 0.0103; wins/losses/ties 2/1/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.3767 +/- 0.2418 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.3409 +/- 0.2052 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.2979 +/- 0.0772 |
| `d18_step2_canonical` | 0.2354 +/- 0.0807 | 0.1481 +/- 0.0259 |  |  | 320.0000 +/- 0.0000 | 1.2872 +/- 0.0253 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0363 +/- 0.0147; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

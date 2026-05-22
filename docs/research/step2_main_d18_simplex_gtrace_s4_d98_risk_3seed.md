# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_simplex.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 1.0405 +/- 0.5570 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 1.0598 +/- 0.4205 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 0.6919 +/- 0.0814 |
| `d18_step2_simplex` | 0.0016 +/- 0.0002 | 0.0025 +/- 0.0002 | 0.9922 +/- 0.0011 | 0.2028 +/- 0.0282 | 320.0000 +/- 0.0000 | 2.5637 +/- 0.1220 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0014 +/- 0.0003; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_simplex': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0371 +/- 0.0102; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.4448 +/- 0.0314 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 0.6676 +/- 0.0890 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.7400 +/- 0.0649 |
| `d18_step2_simplex` | 0.0458 +/- 0.0044 | 0.0619 +/- 0.0028 | 0.7711 +/- 0.0219 | 0.8071 +/- 0.0219 | 320.0000 +/- 0.0000 | 2.4568 +/- 0.1212 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0021 +/- 0.0035; wins/losses/ties 1/2/0; best-D18 counts {'d18_step2_simplex': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0130 +/- 0.0070; wins/losses/ties 0/3/0.

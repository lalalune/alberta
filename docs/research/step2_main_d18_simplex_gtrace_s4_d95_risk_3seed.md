# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_simplex.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 1.0740 +/- 0.4895 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 1.0784 +/- 0.5017 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 0.7596 +/- 0.0944 |
| `d18_step2_simplex` | 0.0016 +/- 0.0002 | 0.0022 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.2010 +/- 0.0268 | 320.0000 +/- 0.0000 | 2.4720 +/- 0.0861 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0014 +/- 0.0003; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_simplex': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0353 +/- 0.0084; wins/losses/ties 3/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 |  | 0.4436 +/- 0.0258 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 |  | 0.6601 +/- 0.0585 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 |  | 0.7272 +/- 0.0539 |
| `d18_step2_simplex` | 0.0507 +/- 0.0045 | 0.0696 +/- 0.0035 | 0.7467 +/- 0.0227 | 0.7928 +/- 0.0310 | 320.0000 +/- 0.0000 | 2.3960 +/- 0.1641 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0028 +/- 0.0037; wins/losses/ties 1/2/0; best-D18 counts {'d18_step2_simplex': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0272 +/- 0.0157; wins/losses/ties 1/2/0.

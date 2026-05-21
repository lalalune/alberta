# D18 Simple Universal Resource-Basis Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0580 +/- 0.0034 | 0.1073 +/- 0.0013 |  |  |  | 0.5652 +/- 0.3145 |
| `mlp_h128` | 0.0751 +/- 0.0086 | 0.1187 +/- 0.0048 |  |  |  | 0.5012 +/- 0.3416 |
| `mlp_h64_64` | 0.0750 +/- 0.0060 | 0.1274 +/- 0.0034 |  |  |  | 0.4297 +/- 0.2240 |
| `d18_step2_canonical` | 0.0174 +/- 0.0018 | 0.0425 +/- 0.0004 |  |  | 320.0000 +/- 0.0000 | 1.8636 +/- 0.3087 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0405 +/- 0.0031; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8611 +/- 0.1893 | 0.9917 +/- 0.0835 |  |  |  | 0.3216 +/- 0.0127 |
| `mlp_h128` | 0.9468 +/- 0.2304 | 1.0382 +/- 0.0979 |  |  |  | 0.2185 +/- 0.0060 |
| `mlp_h64_64` | 0.9490 +/- 0.2284 | 1.0421 +/- 0.1050 |  |  |  | 0.3297 +/- 0.0461 |
| `d18_step2_canonical` | 0.0718 +/- 0.0244 | 0.2462 +/- 0.0340 |  |  | 320.0000 +/- 0.0000 | 1.6303 +/- 0.1181 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.7881 +/- 0.1669; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0732 +/- 0.0114 | 0.1128 +/- 0.0184 |  |  |  | 0.2202 +/- 0.0364 |
| `mlp_h128` | 0.0787 +/- 0.0118 | 0.1165 +/- 0.0180 |  |  |  | 0.2222 +/- 0.0350 |
| `mlp_h64_64` | 0.0933 +/- 0.0081 | 0.1305 +/- 0.0173 |  |  |  | 0.2361 +/- 0.0017 |
| `d18_step2_canonical` | 0.0181 +/- 0.0064 | 0.0518 +/- 0.0119 |  |  | 320.0000 +/- 0.0000 | 1.5538 +/- 0.0752 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0550 +/- 0.0056; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## controlled_triple

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.8637 +/- 0.0489 | 1.0904 +/- 0.0786 |  |  |  | 0.2337 +/- 0.0367 |
| `mlp_h128` | 0.8693 +/- 0.0485 | 1.0951 +/- 0.0761 |  |  |  | 0.1743 +/- 0.0403 |
| `mlp_h64_64` | 0.6059 +/- 0.0368 | 0.9161 +/- 0.0726 |  |  |  | 0.1896 +/- 0.0097 |
| `d18_step2_canonical` | 0.0468 +/- 0.0055 | 0.2399 +/- 0.0173 |  |  | 320.0000 +/- 0.0000 | 1.4419 +/- 0.1031 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.5590 +/- 0.0321; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0053 +/- 0.0003 | 0.0089 +/- 0.0002 | 0.9867 +/- 0.0019 | 0.1348 +/- 0.0272 |  | 1.2916 +/- 0.0323 |
| `mlp_h128` | 0.0077 +/- 0.0002 | 0.0116 +/- 0.0002 | 0.9878 +/- 0.0011 | 0.1528 +/- 0.0172 |  | 0.9718 +/- 0.1989 |
| `mlp_h64_64` | 0.0030 +/- 0.0001 | 0.0064 +/- 0.0001 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |  | 1.0120 +/- 0.0296 |
| `d18_step2_canonical` | 0.0045 +/- 0.0000 | 0.0040 +/- 0.0001 | 0.9889 +/- 0.0011 | 0.2189 +/- 0.0242 | 320.0000 +/- 0.0000 | 4.3159 +/- 0.2723 |

`final_window_mse` best-D18-vs-best-MLP diff: -0.0015 +/- 0.0001; wins/losses/ties 0/3/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0532 +/- 0.0153; wins/losses/ties 3/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0310 +/- 0.0011 | 0.0440 +/- 0.0001 | 0.9133 +/- 0.0084 | 0.9264 +/- 0.0070 |  | 0.8671 +/- 0.0946 |
| `mlp_h128` | 0.0314 +/- 0.0010 | 0.0438 +/- 0.0003 | 0.9089 +/- 0.0056 | 0.9338 +/- 0.0016 |  | 0.7526 +/- 0.1216 |
| `mlp_h64_64` | 0.0332 +/- 0.0017 | 0.0478 +/- 0.0012 | 0.8789 +/- 0.0146 | 0.9109 +/- 0.0057 |  | 0.6272 +/- 0.1384 |
| `d18_step2_canonical` | 0.0199 +/- 0.0007 | 0.0291 +/- 0.0004 | 0.9678 +/- 0.0022 | 0.9685 +/- 0.0039 | 320.0000 +/- 0.0000 | 3.5154 +/- 0.5179 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0111 +/- 0.0005; wins/losses/ties 3/0/0; best-D18 counts {'d18_step2_canonical': 3}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0340 +/- 0.0022; wins/losses/ties 3/0/0.

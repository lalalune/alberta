# D18 Simple Universal Resource-Basis Results

Protocol: 2 paired seeds, 600 online steps, final window 150. Candidate configs: d18_step2_canonical, d18_step2_distilled_memory, d18_step2_distilled_memory_nogates, d18_step2_no_poly.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0983 +/- 0.0116 | 0.1464 +/- 0.0042 |  |  |  | 1.3752 +/- 1.0101 |
| `mlp_h128` | 0.1107 +/- 0.0101 | 0.1570 +/- 0.0009 |  |  |  | 1.3946 +/- 1.0366 |
| `mlp_h64_64` | 0.1203 +/- 0.0130 | 0.1731 +/- 0.0028 |  |  |  | 0.9254 +/- 0.3443 |
| `d18_step2_canonical` | 0.0228 +/- 0.0009 | 0.0673 +/- 0.0015 |  |  | 260.0000 +/- 0.0000 | 3.0689 +/- 0.1530 |
| `d18_step2_distilled_memory` | 0.0222 +/- 0.0007 | 0.0671 +/- 0.0009 |  |  | 260.0000 +/- 0.0000 | 2.5909 +/- 0.1900 |
| `d18_step2_distilled_memory_nogates` | 0.1225 +/- 0.0139 | 0.1727 +/- 0.0109 |  |  | 260.0000 +/- 0.0000 | 3.0165 +/- 0.0698 |
| `d18_step2_no_poly` | 0.0218 +/- 0.0007 | 0.0644 +/- 0.0006 |  |  | 260.0000 +/- 0.0000 | 3.1367 +/- 0.3843 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0765 +/- 0.0110; wins/losses/ties 2/0/0; best-D18 counts {'d18_step2_no_poly': 2}.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0062 +/- 0.0000 | 0.0118 +/- 0.0007 | 0.9867 +/- 0.0000 | 0.1271 +/- 0.0250 |  | 1.0324 +/- 0.1576 |
| `mlp_h128` | 0.0092 +/- 0.0003 | 0.0150 +/- 0.0007 | 0.9833 +/- 0.0033 | 0.1030 +/- 0.0009 |  | 0.5964 +/- 0.0289 |
| `mlp_h64_64` | 0.0041 +/- 0.0002 | 0.0089 +/- 0.0001 | 0.9867 +/- 0.0000 | 0.1002 +/- 0.0019 |  | 0.7525 +/- 0.1178 |
| `d18_step2_canonical` | 0.0027 +/- 0.0000 | 0.0024 +/- 0.0000 | 0.9867 +/- 0.0000 | 0.4249 +/- 0.0278 | 260.0000 +/- 0.0000 | 2.4531 +/- 0.6288 |
| `d18_step2_distilled_memory` | 0.0027 +/- 0.0000 | 0.0022 +/- 0.0001 | 0.9867 +/- 0.0000 | 0.4276 +/- 0.0250 | 260.0000 +/- 0.0000 | 2.1038 +/- 0.4518 |
| `d18_step2_distilled_memory_nogates` | 0.0380 +/- 0.0060 | 0.0241 +/- 0.0010 | 0.8100 +/- 0.0300 | 0.3692 +/- 0.0093 | 258.5000 +/- 0.5000 | 2.2263 +/- 0.3545 |
| `d18_step2_no_poly` | 0.0037 +/- 0.0002 | 0.0034 +/- 0.0001 | 0.9833 +/- 0.0033 | 0.1132 +/- 0.0093 | 260.0000 +/- 0.0000 | 2.4506 +/- 0.9096 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0014 +/- 0.0002; wins/losses/ties 2/0/0; best-D18 counts {'d18_step2_canonical': 2}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.2996 +/- 0.0009; wins/losses/ties 2/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0634 +/- 0.0027 | 0.0698 +/- 0.0015 | 0.6933 +/- 0.0400 | 0.7421 +/- 0.0408 |  | 1.2191 +/- 0.2136 |
| `mlp_h128` | 0.0624 +/- 0.0046 | 0.0701 +/- 0.0023 | 0.7233 +/- 0.0367 | 0.7319 +/- 0.0918 |  | 1.5431 +/- 0.4356 |
| `mlp_h64_64` | 0.0680 +/- 0.0014 | 0.0755 +/- 0.0008 | 0.6433 +/- 0.0100 | 0.6948 +/- 0.0566 |  | 0.8321 +/- 0.2633 |
| `d18_step2_canonical` | 0.0569 +/- 0.0002 | 0.0614 +/- 0.0015 | 0.6700 +/- 0.0100 | 0.7310 +/- 0.1002 | 260.0000 +/- 0.0000 | 5.0588 +/- 0.8462 |
| `d18_step2_distilled_memory` | 0.0569 +/- 0.0006 | 0.0612 +/- 0.0015 | 0.6867 +/- 0.0000 | 0.7338 +/- 0.0863 | 260.0000 +/- 0.0000 | 4.4181 +/- 0.5372 |
| `d18_step2_distilled_memory_nogates` | 0.0520 +/- 0.0107 | 0.0683 +/- 0.0058 | 0.7400 +/- 0.0533 | 0.7894 +/- 0.0046 | 260.0000 +/- 0.0000 | 4.4082 +/- 0.1297 |
| `d18_step2_no_poly` | 0.0565 +/- 0.0005 | 0.0602 +/- 0.0014 | 0.6900 +/- 0.0100 | 0.7161 +/- 0.1002 | 260.0000 +/- 0.0000 | 4.4099 +/- 0.6447 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0102 +/- 0.0062; wins/losses/ties 2/0/0; best-D18 counts {'d18_step2_canonical': 1, 'd18_step2_distilled_memory_nogates': 1}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0455 +/- 0.0380; wins/losses/ties 2/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2513 +/- 0.0358 | 0.2026 +/- 0.0378 |  |  |  | 1.2168 +/- 0.7344 |
| `mlp_h128` | 0.2545 +/- 0.0337 | 0.1970 +/- 0.0346 |  |  |  | 1.0690 +/- 0.4328 |
| `mlp_h64_64` | 0.3124 +/- 0.0535 | 0.2537 +/- 0.0322 |  |  |  | 0.5247 +/- 0.0250 |
| `d18_step2_canonical` | 0.2183 +/- 0.0380 | 0.1747 +/- 0.0330 |  |  | 260.0000 +/- 0.0000 | 3.3525 +/- 0.3234 |
| `d18_step2_distilled_memory` | 0.2285 +/- 0.0302 | 0.1778 +/- 0.0300 |  |  | 260.0000 +/- 0.0000 | 3.8532 +/- 0.9391 |
| `d18_step2_distilled_memory_nogates` | 0.4126 +/- 0.0790 | 0.3211 +/- 0.0629 |  |  | 260.0000 +/- 0.0000 | 3.4938 +/- 0.0180 |
| `d18_step2_no_poly` | 0.2505 +/- 0.0249 | 0.1873 +/- 0.0311 |  |  | 260.0000 +/- 0.0000 | 3.4657 +/- 0.4969 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0330 +/- 0.0022; wins/losses/ties 2/0/0; best-D18 counts {'d18_step2_canonical': 2}.

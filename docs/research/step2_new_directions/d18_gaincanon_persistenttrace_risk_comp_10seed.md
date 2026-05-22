# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_gain_safecore_poly_unified_0p01.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.5580 +/- 0.0902 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.5580 +/- 0.0810 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.5309 +/- 0.0341 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0016 +/- 0.0001 | 0.0021 +/- 0.0001 | 0.9920 +/- 0.0007 | 0.1006 +/- 0.0004 | 320.0000 +/- 0.0000 | 2.3121 +/- 0.0661 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0013 +/- 0.0002; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0353 +/- 0.0119; wins/losses/ties 1/9/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0315 +/- 0.0005 | 0.0452 +/- 0.0004 | 0.9123 +/- 0.0053 | 0.9148 +/- 0.0041 |  | 0.3619 +/- 0.0217 |
| `mlp_h128` | 0.0306 +/- 0.0007 | 0.0438 +/- 0.0003 | 0.9257 +/- 0.0056 | 0.9306 +/- 0.0045 |  | 0.3950 +/- 0.0172 |
| `mlp_h64_64` | 0.0325 +/- 0.0006 | 0.0486 +/- 0.0004 | 0.8900 +/- 0.0058 | 0.9058 +/- 0.0061 |  | 0.4449 +/- 0.0155 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0079 +/- 0.0007 | 0.0200 +/- 0.0004 | 0.9607 +/- 0.0033 | 0.9631 +/- 0.0039 | 320.0000 +/- 0.0000 | 2.0769 +/- 0.0451 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0225 +/- 0.0009; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0321 +/- 0.0036; wins/losses/ties 10/0/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 0.4269 +/- 0.0300 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 0.5349 +/- 0.0337 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 0.5738 +/- 0.0447 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0226 +/- 0.0011 | 0.0439 +/- 0.0009 | 0.8870 +/- 0.0056 | 0.9458 +/- 0.0042 | 320.0000 +/- 0.0000 | 2.2578 +/- 0.0538 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0170 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0458 +/- 0.0057; wins/losses/ties 10/0/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.4721 +/- 0.0190 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.4918 +/- 0.0259 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.5889 +/- 0.0405 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0387 +/- 0.0027 | 0.0556 +/- 0.0020 | 0.8063 +/- 0.0135 | 0.8456 +/- 0.0122 | 320.0000 +/- 0.0000 | 2.3110 +/- 0.0477 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0093 +/- 0.0019; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0096 +/- 0.0093; wins/losses/ties 6/4/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 0.4167 +/- 0.0287 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 0.3668 +/- 0.0119 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 0.4321 +/- 0.0144 |
| `d18_gain_safecore_poly_unified_0p01` | 0.0251 +/- 0.0009 | 0.0408 +/- 0.0004 | 0.8747 +/- 0.0043 | 0.9206 +/- 0.0048 | 320.0000 +/- 0.0000 | 2.2393 +/- 0.0505 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0233 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0425 +/- 0.0051; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2754 +/- 0.0451 | 0.2176 +/- 0.0198 |  |  |  | 0.2805 +/- 0.0420 |
| `mlp_h128` | 0.2731 +/- 0.0453 | 0.2169 +/- 0.0195 |  |  |  | 0.2608 +/- 0.0445 |
| `mlp_h64_64` | 0.3189 +/- 0.0487 | 0.2615 +/- 0.0215 |  |  |  | 0.2974 +/- 0.0233 |
| `d18_gain_safecore_poly_unified_0p01` | 0.2064 +/- 0.0277 | 0.1683 +/- 0.0129 |  |  | 320.0000 +/- 0.0000 | 1.8036 +/- 0.0566 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0629 +/- 0.0210; wins/losses/ties 10/0/0; best-D18 counts {'d18_gain_safecore_poly_unified_0p01': 10}.

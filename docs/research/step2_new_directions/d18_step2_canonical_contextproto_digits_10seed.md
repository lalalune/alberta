# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.2799 +/- 0.0110 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.3018 +/- 0.0083 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.3386 +/- 0.0142 |
| `d18_step2_canonical` | 0.0016 +/- 0.0001 | 0.0021 +/- 0.0001 | 0.9920 +/- 0.0007 | 0.8596 +/- 0.0036 | 320.0000 +/- 0.0000 | 1.8777 +/- 0.0168 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0013 +/- 0.0002; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.7237 +/- 0.0122; wins/losses/ties 10/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0315 +/- 0.0005 | 0.0452 +/- 0.0004 | 0.9123 +/- 0.0053 | 0.9148 +/- 0.0041 |  | 0.3891 +/- 0.0859 |
| `mlp_h128` | 0.0306 +/- 0.0007 | 0.0438 +/- 0.0003 | 0.9257 +/- 0.0056 | 0.9306 +/- 0.0045 |  | 0.3874 +/- 0.0784 |
| `mlp_h64_64` | 0.0325 +/- 0.0006 | 0.0486 +/- 0.0004 | 0.8900 +/- 0.0058 | 0.9058 +/- 0.0061 |  | 0.3479 +/- 0.0139 |
| `d18_step2_canonical` | 0.0079 +/- 0.0007 | 0.0200 +/- 0.0004 | 0.9607 +/- 0.0033 | 0.9269 +/- 0.0105 | 320.0000 +/- 0.0000 | 1.8500 +/- 0.0249 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0225 +/- 0.0009; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0041 +/- 0.0112; wins/losses/ties 4/6/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 0.3895 +/- 0.0276 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 0.3812 +/- 0.0156 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 0.4177 +/- 0.0148 |
| `d18_step2_canonical` | 0.0226 +/- 0.0011 | 0.0439 +/- 0.0009 | 0.8870 +/- 0.0056 | 0.6994 +/- 0.0690 | 320.0000 +/- 0.0000 | 2.1383 +/- 0.0492 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0170 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.2006 +/- 0.0688; wins/losses/ties 4/6/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.3595 +/- 0.0139 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.3902 +/- 0.0116 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.4086 +/- 0.0297 |
| `d18_step2_canonical` | 0.0387 +/- 0.0027 | 0.0556 +/- 0.0020 | 0.8063 +/- 0.0135 | 0.8430 +/- 0.0120 | 320.0000 +/- 0.0000 | 2.1603 +/- 0.0301 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0093 +/- 0.0019; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0071 +/- 0.0092; wins/losses/ties 7/3/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 0.2798 +/- 0.0037 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 0.3188 +/- 0.0164 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 0.3178 +/- 0.0031 |
| `d18_step2_canonical` | 0.0251 +/- 0.0009 | 0.0408 +/- 0.0004 | 0.8747 +/- 0.0043 | 0.8709 +/- 0.0130 | 320.0000 +/- 0.0000 | 1.9114 +/- 0.0191 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0233 +/- 0.0008; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0072 +/- 0.0131; wins/losses/ties 5/5/0.

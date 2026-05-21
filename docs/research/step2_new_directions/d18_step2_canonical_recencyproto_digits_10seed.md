# D18 Simple Universal Resource-Basis Results

Protocol: 10 paired seeds, 1200 online steps, final window 300. Candidate configs: d18_step2_canonical.

Candidate prediction is one additive model: resource-managed RKHS core plus tanh/Fourier and optional finite polynomial bases. There is no output router and no MLP expert.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0051 +/- 0.0002 | 0.0090 +/- 0.0001 | 0.9847 +/- 0.0012 | 0.1197 +/- 0.0089 |  | 0.3122 +/- 0.0160 |
| `mlp_h128` | 0.0074 +/- 0.0002 | 0.0116 +/- 0.0001 | 0.9847 +/- 0.0016 | 0.1301 +/- 0.0109 |  | 0.3444 +/- 0.0144 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9917 +/- 0.0007 | 0.1006 +/- 0.0004 |  | 0.3741 +/- 0.0135 |
| `d18_step2_canonical` | 0.0016 +/- 0.0001 | 0.0021 +/- 0.0001 | 0.9920 +/- 0.0007 | 0.8510 +/- 0.0060 | 320.0000 +/- 0.0000 | 2.0098 +/- 0.0527 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0013 +/- 0.0002; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.7152 +/- 0.0131; wins/losses/ties 10/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0315 +/- 0.0005 | 0.0452 +/- 0.0004 | 0.9123 +/- 0.0053 | 0.9148 +/- 0.0041 |  | 0.3811 +/- 0.0774 |
| `mlp_h128` | 0.0306 +/- 0.0007 | 0.0438 +/- 0.0003 | 0.9257 +/- 0.0056 | 0.9306 +/- 0.0045 |  | 0.4130 +/- 0.0638 |
| `mlp_h64_64` | 0.0325 +/- 0.0006 | 0.0486 +/- 0.0004 | 0.8900 +/- 0.0058 | 0.9058 +/- 0.0061 |  | 0.3877 +/- 0.0328 |
| `d18_step2_canonical` | 0.0152 +/- 0.0005 | 0.0263 +/- 0.0004 | 0.9607 +/- 0.0033 | 0.8991 +/- 0.0047 | 320.0000 +/- 0.0000 | 1.8954 +/- 0.0214 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0152 +/- 0.0007; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0319 +/- 0.0047; wins/losses/ties 0/10/0.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0399 +/- 0.0006 | 0.0570 +/- 0.0006 | 0.8567 +/- 0.0049 | 0.8944 +/- 0.0039 |  | 0.3402 +/- 0.0215 |
| `mlp_h128` | 0.0449 +/- 0.0007 | 0.0598 +/- 0.0004 | 0.8397 +/- 0.0074 | 0.8941 +/- 0.0037 |  | 0.3225 +/- 0.0074 |
| `mlp_h64_64` | 0.0411 +/- 0.0006 | 0.0611 +/- 0.0005 | 0.8400 +/- 0.0060 | 0.8790 +/- 0.0073 |  | 0.3621 +/- 0.0122 |
| `d18_step2_canonical` | 0.0290 +/- 0.0008 | 0.0452 +/- 0.0003 | 0.8870 +/- 0.0056 | 0.8905 +/- 0.0054 | 320.0000 +/- 0.0000 | 2.0480 +/- 0.0938 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0105 +/- 0.0007; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: -0.0095 +/- 0.0072; wins/losses/ties 4/6/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0489 +/- 0.0013 | 0.0602 +/- 0.0007 | 0.7863 +/- 0.0126 | 0.8130 +/- 0.0100 |  | 0.2616 +/- 0.0069 |
| `mlp_h128` | 0.0487 +/- 0.0012 | 0.0596 +/- 0.0008 | 0.8003 +/- 0.0120 | 0.8319 +/- 0.0100 |  | 0.2743 +/- 0.0068 |
| `mlp_h64_64` | 0.0519 +/- 0.0015 | 0.0644 +/- 0.0009 | 0.7487 +/- 0.0109 | 0.8006 +/- 0.0134 |  | 0.3137 +/- 0.0142 |
| `d18_step2_canonical` | 0.0421 +/- 0.0022 | 0.0543 +/- 0.0013 | 0.8063 +/- 0.0135 | 0.8386 +/- 0.0073 | 320.0000 +/- 0.0000 | 1.8253 +/- 0.0180 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0059 +/- 0.0016; wins/losses/ties 8/2/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0026 +/- 0.0074; wins/losses/ties 6/4/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0008 | 0.0611 +/- 0.0003 | 0.8030 +/- 0.0089 | 0.8475 +/- 0.0094 |  | 0.3009 +/- 0.0091 |
| `mlp_h128` | 0.0488 +/- 0.0008 | 0.0598 +/- 0.0002 | 0.8153 +/- 0.0087 | 0.8763 +/- 0.0064 |  | 0.3377 +/- 0.0186 |
| `mlp_h64_64` | 0.0571 +/- 0.0008 | 0.0680 +/- 0.0004 | 0.7373 +/- 0.0106 | 0.8226 +/- 0.0118 |  | 0.3601 +/- 0.0103 |
| `d18_step2_canonical` | 0.0330 +/- 0.0009 | 0.0444 +/- 0.0004 | 0.8747 +/- 0.0043 | 0.8889 +/- 0.0042 | 320.0000 +/- 0.0000 | 2.0068 +/- 0.0463 |

`final_window_mse` best-D18-vs-best-MLP diff: +0.0154 +/- 0.0007; wins/losses/ties 10/0/0; best-D18 counts {'d18_step2_canonical': 10}.
`test_accuracy` best-D18-vs-best-MLP diff: +0.0108 +/- 0.0060; wins/losses/ties 7/3/0.

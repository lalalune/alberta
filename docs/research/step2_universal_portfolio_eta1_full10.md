# Step 2 Universal Portfolio

Protocol: 10 seeds, 1200 steps, final window 300, Hedge eta=1.0, discount=0.995, deployment policy=convex, retention router=class_imbalance, digits deployment objective=mse.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge; the router policies remain available as negative-control variants.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0030 +/- 0.0001 | 0.0045 +/- 0.0000 | 0.9883 +/- 0.0007 | 0.2245 +/- 0.0104 |
| `mlp_h64` | 0.0052 +/- 0.0001 | 0.0091 +/- 0.0001 | 0.9847 +/- 0.0015 | 0.1195 +/- 0.0079 |
| `mlp_h128` | 0.0073 +/- 0.0001 | 0.0115 +/- 0.0001 | 0.9837 +/- 0.0015 | 0.1195 +/- 0.0081 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9913 +/- 0.0005 | 0.1006 +/- 0.0004 |
| `upgd_low_noise` | 0.0208 +/- 0.0007 | 0.0273 +/- 0.0004 | 0.9287 +/- 0.0043 | 0.2245 +/- 0.0104 |
| `dynamic_sparse` | 0.0271 +/- 0.0006 | 0.0397 +/- 0.0005 | 0.9317 +/- 0.0021 | 0.2135 +/- 0.0126 |

`final_window_mse` mixture-vs-best-MLP diff: -0.0001 +/- 0.0000; wins/losses/ties 1/9/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0970 +/- 0.0114; wins/losses/ties 10/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0228 +/- 0.0005 | 0.0339 +/- 0.0003 | 0.9420 +/- 0.0043 | 0.9449 +/- 0.0032 |
| `mlp_h64` | 0.0315 +/- 0.0008 | 0.0449 +/- 0.0005 | 0.9150 +/- 0.0060 | 0.9200 +/- 0.0035 |
| `mlp_h128` | 0.0302 +/- 0.0004 | 0.0433 +/- 0.0004 | 0.9270 +/- 0.0027 | 0.9310 +/- 0.0044 |
| `mlp_h64_64` | 0.0317 +/- 0.0008 | 0.0481 +/- 0.0006 | 0.9083 +/- 0.0069 | 0.9147 +/- 0.0023 |
| `upgd_low_noise` | 0.0316 +/- 0.0006 | 0.0517 +/- 0.0006 | 0.9117 +/- 0.0073 | 0.9180 +/- 0.0054 |
| `dynamic_sparse` | 0.0405 +/- 0.0009 | 0.0681 +/- 0.0007 | 0.8780 +/- 0.0079 | 0.8881 +/- 0.0044 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0071 +/- 0.0001; wins/losses/ties 10/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0128 +/- 0.0023; wins/losses/ties 9/0/1.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0334 +/- 0.0008 | 0.0474 +/- 0.0004 | 0.8873 +/- 0.0064 | 0.9163 +/- 0.0042 |
| `mlp_h64` | 0.0396 +/- 0.0008 | 0.0568 +/- 0.0006 | 0.8623 +/- 0.0075 | 0.8946 +/- 0.0029 |
| `mlp_h128` | 0.0435 +/- 0.0009 | 0.0590 +/- 0.0006 | 0.8447 +/- 0.0080 | 0.8959 +/- 0.0058 |
| `mlp_h64_64` | 0.0407 +/- 0.0011 | 0.0605 +/- 0.0007 | 0.8420 +/- 0.0110 | 0.8870 +/- 0.0053 |
| `upgd_low_noise` | 0.0567 +/- 0.0006 | 0.0745 +/- 0.0007 | 0.7003 +/- 0.0071 | 0.8369 +/- 0.0078 |
| `dynamic_sparse` | 0.0638 +/- 0.0012 | 0.0891 +/- 0.0007 | 0.6693 +/- 0.0155 | 0.8087 +/- 0.0084 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0055 +/- 0.0002; wins/losses/ties 10/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0128 +/- 0.0035; wins/losses/ties 8/2/0.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0385 +/- 0.0012 | 0.0481 +/- 0.0007 | 0.8293 +/- 0.0108 | 0.8616 +/- 0.0099 |
| `mlp_h64` | 0.0493 +/- 0.0016 | 0.0606 +/- 0.0008 | 0.7873 +/- 0.0140 | 0.8174 +/- 0.0130 |
| `mlp_h128` | 0.0483 +/- 0.0012 | 0.0596 +/- 0.0007 | 0.8053 +/- 0.0084 | 0.8453 +/- 0.0109 |
| `mlp_h64_64` | 0.0522 +/- 0.0015 | 0.0646 +/- 0.0011 | 0.7570 +/- 0.0132 | 0.7998 +/- 0.0132 |
| `upgd_low_noise` | 0.0471 +/- 0.0013 | 0.0631 +/- 0.0005 | 0.7987 +/- 0.0146 | 0.8275 +/- 0.0129 |
| `dynamic_sparse` | 0.0586 +/- 0.0014 | 0.0797 +/- 0.0009 | 0.7337 +/- 0.0124 | 0.7857 +/- 0.0113 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0094 +/- 0.0002; wins/losses/ties 10/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0158 +/- 0.0054; wins/losses/ties 7/3/0.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0407 +/- 0.0007 | 0.0492 +/- 0.0003 | 0.8517 +/- 0.0081 | 0.8931 +/- 0.0045 |
| `mlp_h64` | 0.0506 +/- 0.0009 | 0.0615 +/- 0.0005 | 0.7993 +/- 0.0082 | 0.8525 +/- 0.0061 |
| `mlp_h128` | 0.0484 +/- 0.0011 | 0.0594 +/- 0.0004 | 0.8233 +/- 0.0094 | 0.8755 +/- 0.0053 |
| `mlp_h64_64` | 0.0561 +/- 0.0008 | 0.0674 +/- 0.0005 | 0.7440 +/- 0.0112 | 0.8111 +/- 0.0111 |
| `upgd_low_noise` | 0.0579 +/- 0.0007 | 0.0743 +/- 0.0007 | 0.7067 +/- 0.0087 | 0.8163 +/- 0.0113 |
| `dynamic_sparse` | 0.0662 +/- 0.0009 | 0.0906 +/- 0.0007 | 0.6610 +/- 0.0062 | 0.7900 +/- 0.0136 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0075 +/- 0.0004; wins/losses/ties 10/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0156 +/- 0.0027; wins/losses/ties 9/0/1.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.2706 +/- 0.0446 | 0.2124 +/- 0.0196 |  |  |
| `mlp_h64` | 0.2780 +/- 0.0456 | 0.2200 +/- 0.0199 |  |  |
| `mlp_h128` | 0.2733 +/- 0.0445 | 0.2174 +/- 0.0195 |  |  |
| `mlp_h64_64` | 0.3280 +/- 0.0513 | 0.2660 +/- 0.0239 |  |  |
| `upgd_low_noise` | 0.5051 +/- 0.1199 | 0.3650 +/- 0.0512 |  |  |
| `dynamic_sparse` | 0.4722 +/- 0.1073 | 0.3510 +/- 0.0479 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0018 +/- 0.0008; wins/losses/ties 7/3/0.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 1.5607 +/- 0.3549 | 1.3665 +/- 0.1371 |  |  |
| `mlp_h64` | 1.6235 +/- 0.3676 | 1.4328 +/- 0.1476 |  |  |
| `mlp_h128` | 1.6309 +/- 0.3676 | 1.4433 +/- 0.1496 |  |  |
| `mlp_h64_64` | 1.5961 +/- 0.3552 | 1.4058 +/- 0.1405 |  |  |
| `upgd_low_noise` | 1.6327 +/- 0.3528 | 1.4118 +/- 0.1405 |  |  |
| `dynamic_sparse` | 1.6312 +/- 0.3568 | 1.4115 +/- 0.1423 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0168 +/- 0.0128; wins/losses/ties 6/4/0.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.9606 +/- 0.2077 | 0.9429 +/- 0.0921 |  |  |
| `mlp_h64` | 1.0311 +/- 0.2152 | 1.0141 +/- 0.0964 |  |  |
| `mlp_h128` | 1.0409 +/- 0.2160 | 1.0186 +/- 0.0965 |  |  |
| `mlp_h64_64` | 1.0020 +/- 0.2117 | 0.9976 +/- 0.0957 |  |  |
| `upgd_low_noise` | 0.9621 +/- 0.2081 | 0.9434 +/- 0.0921 |  |  |
| `dynamic_sparse` | 0.9652 +/- 0.2075 | 0.9507 +/- 0.0914 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0414 +/- 0.0072; wins/losses/ties 10/0/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

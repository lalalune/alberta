# Step 2 Universal Portfolio

Protocol: 30 seeds, 1200 steps, final window 300, Hedge eta=1.0, discount=0.995, deployment policy=convex, retention router=class_imbalance, digits deployment objective=mse.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge with a digits class-imbalance online MSE guard; the router policies remain available as negative-control variants.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0232 +/- 0.0003 | 0.0340 +/- 0.0001 | 0.9354 +/- 0.0023 | 0.9430 +/- 0.0017 |
| `mlp_h64` | 0.0319 +/- 0.0004 | 0.0447 +/- 0.0003 | 0.9087 +/- 0.0033 | 0.9221 +/- 0.0022 |
| `mlp_h128` | 0.0308 +/- 0.0003 | 0.0437 +/- 0.0002 | 0.9196 +/- 0.0023 | 0.9307 +/- 0.0023 |
| `mlp_h64_64` | 0.0323 +/- 0.0004 | 0.0487 +/- 0.0004 | 0.8980 +/- 0.0041 | 0.9066 +/- 0.0028 |
| `upgd_low_noise` | 0.0319 +/- 0.0003 | 0.0509 +/- 0.0004 | 0.9059 +/- 0.0036 | 0.9178 +/- 0.0027 |
| `dynamic_sparse` | 0.0397 +/- 0.0004 | 0.0676 +/- 0.0004 | 0.8816 +/- 0.0038 | 0.8944 +/- 0.0023 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0073 +/- 0.0001; wins/losses/ties 30/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0106 +/- 0.0013; wins/losses/ties 26/2/2.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0335 +/- 0.0004 | 0.0470 +/- 0.0003 | 0.8819 +/- 0.0033 | 0.9153 +/- 0.0027 |
| `mlp_h64` | 0.0396 +/- 0.0004 | 0.0561 +/- 0.0003 | 0.8602 +/- 0.0037 | 0.8974 +/- 0.0030 |
| `mlp_h128` | 0.0439 +/- 0.0006 | 0.0587 +/- 0.0004 | 0.8386 +/- 0.0049 | 0.8967 +/- 0.0029 |
| `mlp_h64_64` | 0.0408 +/- 0.0006 | 0.0606 +/- 0.0004 | 0.8420 +/- 0.0055 | 0.8854 +/- 0.0033 |
| `upgd_low_noise` | 0.0554 +/- 0.0006 | 0.0729 +/- 0.0005 | 0.7101 +/- 0.0069 | 0.8408 +/- 0.0052 |
| `dynamic_sparse` | 0.0625 +/- 0.0008 | 0.0880 +/- 0.0005 | 0.6833 +/- 0.0093 | 0.8145 +/- 0.0062 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0056 +/- 0.0001; wins/losses/ties 30/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0116 +/- 0.0019; wins/losses/ties 25/4/1.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0392 +/- 0.0007 | 0.0482 +/- 0.0003 | 0.8264 +/- 0.0069 | 0.8545 +/- 0.0066 |
| `mlp_h64` | 0.0499 +/- 0.0009 | 0.0604 +/- 0.0004 | 0.7818 +/- 0.0080 | 0.8173 +/- 0.0076 |
| `mlp_h128` | 0.0490 +/- 0.0008 | 0.0596 +/- 0.0004 | 0.8013 +/- 0.0068 | 0.8370 +/- 0.0070 |
| `mlp_h64_64` | 0.0526 +/- 0.0009 | 0.0651 +/- 0.0005 | 0.7532 +/- 0.0078 | 0.8004 +/- 0.0078 |
| `upgd_low_noise` | 0.0480 +/- 0.0007 | 0.0630 +/- 0.0004 | 0.7906 +/- 0.0077 | 0.8233 +/- 0.0071 |
| `dynamic_sparse` | 0.0580 +/- 0.0007 | 0.0794 +/- 0.0004 | 0.7342 +/- 0.0069 | 0.7874 +/- 0.0082 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0094 +/- 0.0002; wins/losses/ties 30/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0169 +/- 0.0025; wins/losses/ties 25/4/1.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0407 +/- 0.0004 | 0.0491 +/- 0.0002 | 0.8418 +/- 0.0046 | 0.8962 +/- 0.0028 |
| `mlp_h64` | 0.0502 +/- 0.0005 | 0.0608 +/- 0.0003 | 0.7970 +/- 0.0054 | 0.8561 +/- 0.0040 |
| `mlp_h128` | 0.0484 +/- 0.0005 | 0.0595 +/- 0.0002 | 0.8211 +/- 0.0044 | 0.8802 +/- 0.0033 |
| `mlp_h64_64` | 0.0566 +/- 0.0005 | 0.0679 +/- 0.0004 | 0.7387 +/- 0.0054 | 0.8190 +/- 0.0050 |
| `upgd_low_noise` | 0.0580 +/- 0.0004 | 0.0732 +/- 0.0004 | 0.7074 +/- 0.0048 | 0.8241 +/- 0.0053 |
| `dynamic_sparse` | 0.0666 +/- 0.0005 | 0.0912 +/- 0.0004 | 0.6617 +/- 0.0056 | 0.7867 +/- 0.0053 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0075 +/- 0.0002; wins/losses/ties 30/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0152 +/- 0.0019; wins/losses/ties 27/2/1.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

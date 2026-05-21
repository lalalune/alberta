# Step 2 Universal Portfolio

Protocol: 3 seeds, 1200 steps, final window 300, Hedge eta=8.0, discount=0.995, deployment policy=convex, retention router=class_imbalance, digits deployment objective=mlp_h128.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge; the router policies remain available as negative-control variants.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0028 +/- 0.0002 | 0.0056 +/- 0.0002 | 0.9922 +/- 0.0011 | 0.2307 +/- 0.0172 |
| `mlp_h64` | 0.0052 +/- 0.0002 | 0.0091 +/- 0.0001 | 0.9878 +/- 0.0040 | 0.1367 +/- 0.0204 |
| `mlp_h128` | 0.0073 +/- 0.0001 | 0.0115 +/- 0.0001 | 0.9867 +/- 0.0019 | 0.1484 +/- 0.0168 |
| `mlp_h64_64` | 0.0028 +/- 0.0002 | 0.0061 +/- 0.0000 | 0.9922 +/- 0.0011 | 0.1002 +/- 0.0000 |
| `upgd_low_noise` | 0.0200 +/- 0.0009 | 0.0267 +/- 0.0006 | 0.9311 +/- 0.0091 | 0.2307 +/- 0.0172 |
| `dynamic_sparse` | 0.0274 +/- 0.0002 | 0.0393 +/- 0.0003 | 0.9333 +/- 0.0033 | 0.2103 +/- 0.0136 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0000 +/- 0.0000; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0674 +/- 0.0125; wins/losses/ties 3/0/0.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0283 +/- 0.0018 | 0.0398 +/- 0.0010 | 0.9289 +/- 0.0040 | 0.9412 +/- 0.0069 |
| `mlp_h64` | 0.0317 +/- 0.0018 | 0.0441 +/- 0.0010 | 0.9144 +/- 0.0128 | 0.9221 +/- 0.0037 |
| `mlp_h128` | 0.0298 +/- 0.0011 | 0.0421 +/- 0.0003 | 0.9289 +/- 0.0056 | 0.9412 +/- 0.0069 |
| `mlp_h64_64` | 0.0326 +/- 0.0018 | 0.0486 +/- 0.0006 | 0.8944 +/- 0.0109 | 0.9128 +/- 0.0077 |
| `upgd_low_noise` | 0.0323 +/- 0.0014 | 0.0509 +/- 0.0003 | 0.9000 +/- 0.0204 | 0.9171 +/- 0.0115 |
| `dynamic_sparse` | 0.0413 +/- 0.0023 | 0.0691 +/- 0.0001 | 0.8689 +/- 0.0230 | 0.8782 +/- 0.0048 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0016 +/- 0.0007; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0000 +/- 0.0000; wins/losses/ties 0/0/3.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0386 +/- 0.0020 | 0.0529 +/- 0.0014 | 0.8700 +/- 0.0150 | 0.9054 +/- 0.0156 |
| `mlp_h64` | 0.0387 +/- 0.0020 | 0.0556 +/- 0.0013 | 0.8689 +/- 0.0144 | 0.9029 +/- 0.0041 |
| `mlp_h128` | 0.0425 +/- 0.0008 | 0.0577 +/- 0.0009 | 0.8522 +/- 0.0029 | 0.9054 +/- 0.0156 |
| `mlp_h64_64` | 0.0412 +/- 0.0017 | 0.0612 +/- 0.0009 | 0.8289 +/- 0.0111 | 0.8924 +/- 0.0170 |
| `upgd_low_noise` | 0.0570 +/- 0.0004 | 0.0739 +/- 0.0007 | 0.6900 +/- 0.0069 | 0.8404 +/- 0.0239 |
| `dynamic_sparse` | 0.0638 +/- 0.0013 | 0.0899 +/- 0.0006 | 0.6600 +/- 0.0067 | 0.8176 +/- 0.0183 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0001 +/- 0.0001; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: -0.0093 +/- 0.0060; wins/losses/ties 0/2/1.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0450 +/- 0.0005 | 0.0554 +/- 0.0014 | 0.8011 +/- 0.0097 | 0.8194 +/- 0.0192 |
| `mlp_h64` | 0.0514 +/- 0.0012 | 0.0609 +/- 0.0012 | 0.7789 +/- 0.0232 | 0.7811 +/- 0.0039 |
| `mlp_h128` | 0.0493 +/- 0.0012 | 0.0598 +/- 0.0006 | 0.7878 +/- 0.0166 | 0.8194 +/- 0.0192 |
| `mlp_h64_64` | 0.0535 +/- 0.0008 | 0.0650 +/- 0.0017 | 0.7433 +/- 0.0077 | 0.7730 +/- 0.0080 |
| `upgd_low_noise` | 0.0490 +/- 0.0016 | 0.0640 +/- 0.0003 | 0.7878 +/- 0.0244 | 0.8275 +/- 0.0195 |
| `dynamic_sparse` | 0.0596 +/- 0.0014 | 0.0805 +/- 0.0011 | 0.7456 +/- 0.0146 | 0.7792 +/- 0.0093 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0040 +/- 0.0007; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0000 +/- 0.0000; wins/losses/ties 0/0/3.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0470 +/- 0.0029 | 0.0556 +/- 0.0009 | 0.8189 +/- 0.0278 | 0.8658 +/- 0.0139 |
| `mlp_h64` | 0.0498 +/- 0.0024 | 0.0606 +/- 0.0009 | 0.8067 +/- 0.0236 | 0.8578 +/- 0.0110 |
| `mlp_h128` | 0.0500 +/- 0.0032 | 0.0590 +/- 0.0011 | 0.8144 +/- 0.0306 | 0.8658 +/- 0.0139 |
| `mlp_h64_64` | 0.0577 +/- 0.0014 | 0.0685 +/- 0.0007 | 0.7133 +/- 0.0171 | 0.7817 +/- 0.0199 |
| `upgd_low_noise` | 0.0572 +/- 0.0013 | 0.0729 +/- 0.0010 | 0.7167 +/- 0.0164 | 0.8318 +/- 0.0193 |
| `dynamic_sparse` | 0.0659 +/- 0.0012 | 0.0913 +/- 0.0003 | 0.6611 +/- 0.0164 | 0.8027 +/- 0.0241 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0022 +/- 0.0012; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: -0.0068 +/- 0.0068; wins/losses/ties 0/1/2.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

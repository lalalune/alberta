# Step 2 Universal Portfolio

Protocol: 3 seeds, 1200 steps, final window 300, Hedge eta=8.0, discount=0.995, retention router=class_imbalance.

The live portfolio keeps two causal routers: MLP-only Hedge as a fallback and all-expert Hedge over `mlp_h64`, `mlp_h128`, `mlp_h64_64`, `upgd_low_noise`, and `dynamic_sparse`. A causal EMA guard deploys the all-expert prediction only when its prior loss trace does not trail the MLP-only fallback.

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
| `mixture` | 0.0284 +/- 0.0018 | 0.0398 +/- 0.0010 | 0.9278 +/- 0.0044 | 0.9412 +/- 0.0069 |
| `mlp_h64` | 0.0317 +/- 0.0018 | 0.0441 +/- 0.0010 | 0.9144 +/- 0.0128 | 0.9221 +/- 0.0037 |
| `mlp_h128` | 0.0298 +/- 0.0011 | 0.0421 +/- 0.0003 | 0.9289 +/- 0.0056 | 0.9412 +/- 0.0069 |
| `mlp_h64_64` | 0.0326 +/- 0.0018 | 0.0486 +/- 0.0006 | 0.8944 +/- 0.0109 | 0.9128 +/- 0.0077 |
| `upgd_low_noise` | 0.0323 +/- 0.0014 | 0.0509 +/- 0.0003 | 0.9000 +/- 0.0204 | 0.9171 +/- 0.0115 |
| `dynamic_sparse` | 0.0413 +/- 0.0023 | 0.0691 +/- 0.0001 | 0.8689 +/- 0.0230 | 0.8782 +/- 0.0048 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0014 +/- 0.0008; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0000 +/- 0.0000; wins/losses/ties 0/0/3.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0386 +/- 0.0020 | 0.0529 +/- 0.0014 | 0.8700 +/- 0.0150 | 0.9048 +/- 0.0041 |
| `mlp_h64` | 0.0387 +/- 0.0020 | 0.0556 +/- 0.0013 | 0.8689 +/- 0.0144 | 0.9029 +/- 0.0041 |
| `mlp_h128` | 0.0425 +/- 0.0008 | 0.0577 +/- 0.0009 | 0.8522 +/- 0.0029 | 0.9054 +/- 0.0156 |
| `mlp_h64_64` | 0.0412 +/- 0.0017 | 0.0612 +/- 0.0009 | 0.8289 +/- 0.0111 | 0.8924 +/- 0.0170 |
| `upgd_low_noise` | 0.0570 +/- 0.0004 | 0.0739 +/- 0.0007 | 0.6900 +/- 0.0069 | 0.8404 +/- 0.0239 |
| `dynamic_sparse` | 0.0638 +/- 0.0013 | 0.0899 +/- 0.0006 | 0.6600 +/- 0.0067 | 0.8176 +/- 0.0183 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0001 +/- 0.0001; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: -0.0099 +/- 0.0108; wins/losses/ties 1/1/1.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0472 +/- 0.0003 | 0.0562 +/- 0.0014 | 0.7944 +/- 0.0118 | 0.8169 +/- 0.0164 |
| `mlp_h64` | 0.0514 +/- 0.0012 | 0.0609 +/- 0.0012 | 0.7789 +/- 0.0232 | 0.7811 +/- 0.0039 |
| `mlp_h128` | 0.0493 +/- 0.0012 | 0.0598 +/- 0.0006 | 0.7878 +/- 0.0166 | 0.8194 +/- 0.0192 |
| `mlp_h64_64` | 0.0535 +/- 0.0008 | 0.0650 +/- 0.0017 | 0.7433 +/- 0.0077 | 0.7730 +/- 0.0080 |
| `upgd_low_noise` | 0.0490 +/- 0.0016 | 0.0640 +/- 0.0003 | 0.7878 +/- 0.0244 | 0.8275 +/- 0.0195 |
| `dynamic_sparse` | 0.0596 +/- 0.0014 | 0.0805 +/- 0.0011 | 0.7456 +/- 0.0146 | 0.7792 +/- 0.0093 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0018 +/- 0.0010; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: -0.0025 +/- 0.0034; wins/losses/ties 1/1/1.

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0470 +/- 0.0030 | 0.0556 +/- 0.0009 | 0.8189 +/- 0.0278 | 0.8707 +/- 0.0104 |
| `mlp_h64` | 0.0498 +/- 0.0024 | 0.0606 +/- 0.0009 | 0.8067 +/- 0.0236 | 0.8578 +/- 0.0110 |
| `mlp_h128` | 0.0500 +/- 0.0032 | 0.0590 +/- 0.0011 | 0.8144 +/- 0.0306 | 0.8658 +/- 0.0139 |
| `mlp_h64_64` | 0.0577 +/- 0.0014 | 0.0685 +/- 0.0007 | 0.7133 +/- 0.0171 | 0.7817 +/- 0.0199 |
| `upgd_low_noise` | 0.0572 +/- 0.0013 | 0.0729 +/- 0.0010 | 0.7167 +/- 0.0164 | 0.8318 +/- 0.0193 |
| `dynamic_sparse` | 0.0659 +/- 0.0012 | 0.0913 +/- 0.0003 | 0.6611 +/- 0.0164 | 0.8027 +/- 0.0241 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0021 +/- 0.0012; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: -0.0019 +/- 0.0039; wins/losses/ties 1/1/1.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.2762 +/- 0.0929 | 0.1790 +/- 0.0292 |  |  |
| `mlp_h64` | 0.2759 +/- 0.0935 | 0.1832 +/- 0.0298 |  |  |
| `mlp_h128` | 0.2781 +/- 0.0926 | 0.1797 +/- 0.0286 |  |  |
| `mlp_h64_64` | 0.3250 +/- 0.1132 | 0.2192 +/- 0.0368 |  |  |
| `upgd_low_noise` | 0.4057 +/- 0.1466 | 0.2545 +/- 0.0513 |  |  |
| `dynamic_sparse` | 0.3989 +/- 0.1519 | 0.2557 +/- 0.0518 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: -0.0011 +/- 0.0012; wins/losses/ties 2/1/0.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 1.1139 +/- 0.2883 | 1.2847 +/- 0.2353 |  |  |
| `mlp_h64` | 1.1486 +/- 0.2623 | 1.3420 +/- 0.2454 |  |  |
| `mlp_h128` | 1.1763 +/- 0.2476 | 1.3586 +/- 0.2511 |  |  |
| `mlp_h64_64` | 1.1346 +/- 0.2865 | 1.3197 +/- 0.2280 |  |  |
| `upgd_low_noise` | 1.2632 +/- 0.1740 | 1.3579 +/- 0.2781 |  |  |
| `dynamic_sparse` | 1.2448 +/- 0.1793 | 1.3506 +/- 0.2596 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0128 +/- 0.0298; wins/losses/ties 1/2/0.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.9182 +/- 0.6380 | 0.9811 +/- 0.2289 |  |  |
| `mlp_h64` | 0.9667 +/- 0.6508 | 1.0398 +/- 0.2380 |  |  |
| `mlp_h128` | 0.9740 +/- 0.6530 | 1.0456 +/- 0.2395 |  |  |
| `mlp_h64_64` | 0.9497 +/- 0.6436 | 1.0271 +/- 0.2363 |  |  |
| `upgd_low_noise` | 0.9132 +/- 0.6341 | 0.9766 +/- 0.2277 |  |  |
| `dynamic_sparse` | 0.9204 +/- 0.6367 | 0.9789 +/- 0.2263 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0314 +/- 0.0074; wins/losses/ties 3/0/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

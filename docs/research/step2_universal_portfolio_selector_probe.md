# Step 2 Universal Portfolio

Protocol: 3 seeds, 1200 steps, final window 300, Hedge eta=8.0, discount=0.995, retention router=class_imbalance.

The live portfolio keeps two causal routers: MLP-only Hedge as a fallback and all-expert Hedge over `mlp_h64`, `mlp_h128`, `mlp_h64_64`, `upgd_low_noise`, and `dynamic_sparse`. Hedge learns weights, but the deployed online prediction is winner-take-all from the current top-weight route; a causal EMA guard uses the all-expert selector only when its prior loss trace does not trail the MLP-only selector.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0299 +/- 0.0010 | 0.0422 +/- 0.0003 | 0.9289 +/- 0.0056 | 0.9344 +/- 0.0077 |
| `mlp_h64` | 0.0317 +/- 0.0018 | 0.0441 +/- 0.0010 | 0.9144 +/- 0.0128 | 0.9221 +/- 0.0037 |
| `mlp_h128` | 0.0298 +/- 0.0011 | 0.0421 +/- 0.0003 | 0.9289 +/- 0.0056 | 0.9412 +/- 0.0069 |
| `mlp_h64_64` | 0.0326 +/- 0.0018 | 0.0486 +/- 0.0006 | 0.8944 +/- 0.0109 | 0.9128 +/- 0.0077 |
| `upgd_low_noise` | 0.0323 +/- 0.0014 | 0.0509 +/- 0.0003 | 0.9000 +/- 0.0204 | 0.9171 +/- 0.0115 |
| `dynamic_sparse` | 0.0413 +/- 0.0023 | 0.0691 +/- 0.0001 | 0.8689 +/- 0.0230 | 0.8782 +/- 0.0048 |

`final_window_mse` mixture-vs-best-MLP diff: -0.0001 +/- 0.0001; wins/losses/ties 0/1/2.
`test_accuracy` mixture-vs-best-MLP diff: -0.0068 +/- 0.0068; wins/losses/ties 0/1/2.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0486 +/- 0.0010 | 0.0598 +/- 0.0007 | 0.7911 +/- 0.0113 | 0.8182 +/- 0.0204 |
| `mlp_h64` | 0.0514 +/- 0.0012 | 0.0609 +/- 0.0012 | 0.7789 +/- 0.0232 | 0.7811 +/- 0.0039 |
| `mlp_h128` | 0.0493 +/- 0.0012 | 0.0598 +/- 0.0006 | 0.7878 +/- 0.0166 | 0.8194 +/- 0.0192 |
| `mlp_h64_64` | 0.0535 +/- 0.0008 | 0.0650 +/- 0.0017 | 0.7433 +/- 0.0077 | 0.7730 +/- 0.0080 |
| `upgd_low_noise` | 0.0490 +/- 0.0016 | 0.0640 +/- 0.0003 | 0.7878 +/- 0.0244 | 0.8275 +/- 0.0195 |
| `dynamic_sparse` | 0.0596 +/- 0.0014 | 0.0805 +/- 0.0011 | 0.7456 +/- 0.0146 | 0.7792 +/- 0.0093 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0005 +/- 0.0004; wins/losses/ties 2/1/0.
`test_accuracy` mixture-vs-best-MLP diff: -0.0012 +/- 0.0012; wins/losses/ties 0/1/2.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.2769 +/- 0.0934 | 0.1797 +/- 0.0294 |  |  |
| `mlp_h64` | 0.2759 +/- 0.0935 | 0.1832 +/- 0.0298 |  |  |
| `mlp_h128` | 0.2781 +/- 0.0926 | 0.1797 +/- 0.0286 |  |  |
| `mlp_h64_64` | 0.3250 +/- 0.1132 | 0.2192 +/- 0.0368 |  |  |
| `upgd_low_noise` | 0.4057 +/- 0.1466 | 0.2545 +/- 0.0513 |  |  |
| `dynamic_sparse` | 0.3989 +/- 0.1519 | 0.2557 +/- 0.0518 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: -0.0018 +/- 0.0011; wins/losses/ties 0/2/1.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 1.1159 +/- 0.2895 | 1.2848 +/- 0.2349 |  |  |
| `mlp_h64` | 1.1486 +/- 0.2623 | 1.3420 +/- 0.2454 |  |  |
| `mlp_h128` | 1.1763 +/- 0.2476 | 1.3586 +/- 0.2511 |  |  |
| `mlp_h64_64` | 1.1346 +/- 0.2865 | 1.3197 +/- 0.2280 |  |  |
| `upgd_low_noise` | 1.2632 +/- 0.1740 | 1.3579 +/- 0.2781 |  |  |
| `dynamic_sparse` | 1.2448 +/- 0.1793 | 1.3506 +/- 0.2596 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0108 +/- 0.0292; wins/losses/ties 1/2/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

# Step 2 Universal Portfolio

Protocol: 3 seeds, 1200 steps, final window 300, Hedge eta=8.0, discount=0.995, retention router=class_imbalance.

The live portfolio keeps a causal router-of-routers over four deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. Router selection uses only prior EMA losses.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0292 +/- 0.0019 | 0.0407 +/- 0.0011 | 0.9200 +/- 0.0051 | 0.9388 +/- 0.0043 |
| `mlp_h64` | 0.0317 +/- 0.0018 | 0.0441 +/- 0.0010 | 0.9144 +/- 0.0128 | 0.9221 +/- 0.0037 |
| `mlp_h128` | 0.0298 +/- 0.0011 | 0.0421 +/- 0.0003 | 0.9289 +/- 0.0056 | 0.9412 +/- 0.0069 |
| `mlp_h64_64` | 0.0326 +/- 0.0018 | 0.0486 +/- 0.0006 | 0.8944 +/- 0.0109 | 0.9128 +/- 0.0077 |
| `upgd_low_noise` | 0.0323 +/- 0.0014 | 0.0509 +/- 0.0003 | 0.9000 +/- 0.0204 | 0.9171 +/- 0.0115 |
| `dynamic_sparse` | 0.0413 +/- 0.0023 | 0.0691 +/- 0.0001 | 0.8689 +/- 0.0230 | 0.8782 +/- 0.0048 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0007 +/- 0.0009; wins/losses/ties 2/1/0.
`test_accuracy` mixture-vs-best-MLP diff: -0.0025 +/- 0.0045; wins/losses/ties 1/1/1.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0465 +/- 0.0001 | 0.0570 +/- 0.0016 | 0.7967 +/- 0.0133 | 0.8281 +/- 0.0226 |
| `mlp_h64` | 0.0514 +/- 0.0012 | 0.0609 +/- 0.0012 | 0.7789 +/- 0.0232 | 0.7811 +/- 0.0039 |
| `mlp_h128` | 0.0493 +/- 0.0012 | 0.0598 +/- 0.0006 | 0.7878 +/- 0.0166 | 0.8194 +/- 0.0192 |
| `mlp_h64_64` | 0.0535 +/- 0.0008 | 0.0650 +/- 0.0017 | 0.7433 +/- 0.0077 | 0.7730 +/- 0.0080 |
| `upgd_low_noise` | 0.0490 +/- 0.0016 | 0.0640 +/- 0.0003 | 0.7878 +/- 0.0244 | 0.8275 +/- 0.0195 |
| `dynamic_sparse` | 0.0596 +/- 0.0014 | 0.0805 +/- 0.0011 | 0.7456 +/- 0.0146 | 0.7792 +/- 0.0093 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0026 +/- 0.0009; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0087 +/- 0.0128; wins/losses/ties 2/1/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.2786 +/- 0.0917 | 0.1798 +/- 0.0294 |  |  |
| `mlp_h64` | 0.2759 +/- 0.0935 | 0.1832 +/- 0.0298 |  |  |
| `mlp_h128` | 0.2781 +/- 0.0926 | 0.1797 +/- 0.0286 |  |  |
| `mlp_h64_64` | 0.3250 +/- 0.1132 | 0.2192 +/- 0.0368 |  |  |
| `upgd_low_noise` | 0.4057 +/- 0.1466 | 0.2545 +/- 0.0513 |  |  |
| `dynamic_sparse` | 0.3989 +/- 0.1519 | 0.2557 +/- 0.0518 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: -0.0035 +/- 0.0020; wins/losses/ties 1/2/0.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 1.1384 +/- 0.2957 | 1.2913 +/- 0.2302 |  |  |
| `mlp_h64` | 1.1486 +/- 0.2623 | 1.3420 +/- 0.2454 |  |  |
| `mlp_h128` | 1.1763 +/- 0.2476 | 1.3586 +/- 0.2511 |  |  |
| `mlp_h64_64` | 1.1346 +/- 0.2865 | 1.3197 +/- 0.2280 |  |  |
| `upgd_low_noise` | 1.2632 +/- 0.1740 | 1.3579 +/- 0.2781 |  |  |
| `dynamic_sparse` | 1.2448 +/- 0.1793 | 1.3506 +/- 0.2596 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: -0.0117 +/- 0.0292; wins/losses/ties 1/2/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

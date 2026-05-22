# Step 2 Universal Portfolio

Protocol: 10 seeds, 1200 steps, final window 300, Hedge eta=1.0, discount=0.995, deployment policy=guarded_best_mlp, retention router=class_imbalance, digits deployment objective=mse.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge; the router policies remain available as negative-control variants.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0029 +/- 0.0001 | 0.0045 +/- 0.0001 | 0.9883 +/- 0.0007 | 0.2245 +/- 0.0104 |
| `mlp_h64` | 0.0052 +/- 0.0001 | 0.0091 +/- 0.0001 | 0.9847 +/- 0.0015 | 0.1195 +/- 0.0079 |
| `mlp_h128` | 0.0073 +/- 0.0001 | 0.0115 +/- 0.0001 | 0.9837 +/- 0.0015 | 0.1195 +/- 0.0081 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9913 +/- 0.0005 | 0.1006 +/- 0.0004 |
| `upgd_low_noise` | 0.0208 +/- 0.0007 | 0.0273 +/- 0.0004 | 0.9287 +/- 0.0043 | 0.2245 +/- 0.0104 |
| `dynamic_sparse` | 0.0271 +/- 0.0006 | 0.0397 +/- 0.0005 | 0.9317 +/- 0.0021 | 0.2135 +/- 0.0126 |

`final_window_mse` mixture-vs-best-MLP diff: -0.0000 +/- 0.0000; wins/losses/ties 3/7/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0970 +/- 0.0114; wins/losses/ties 10/0/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.2712 +/- 0.0448 | 0.2126 +/- 0.0195 |  |  |
| `mlp_h64` | 0.2780 +/- 0.0456 | 0.2200 +/- 0.0199 |  |  |
| `mlp_h128` | 0.2733 +/- 0.0445 | 0.2174 +/- 0.0195 |  |  |
| `mlp_h64_64` | 0.3280 +/- 0.0513 | 0.2660 +/- 0.0239 |  |  |
| `upgd_low_noise` | 0.5051 +/- 0.1199 | 0.3650 +/- 0.0512 |  |  |
| `dynamic_sparse` | 0.4722 +/- 0.1073 | 0.3510 +/- 0.0479 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0011 +/- 0.0007; wins/losses/ties 7/3/0.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 1.5644 +/- 0.3580 | 1.3688 +/- 0.1376 |  |  |
| `mlp_h64` | 1.6235 +/- 0.3676 | 1.4328 +/- 0.1476 |  |  |
| `mlp_h128` | 1.6309 +/- 0.3676 | 1.4433 +/- 0.1496 |  |  |
| `mlp_h64_64` | 1.5961 +/- 0.3552 | 1.4058 +/- 0.1405 |  |  |
| `upgd_low_noise` | 1.6327 +/- 0.3528 | 1.4118 +/- 0.1405 |  |  |
| `dynamic_sparse` | 1.6312 +/- 0.3568 | 1.4115 +/- 0.1423 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0131 +/- 0.0110; wins/losses/ties 5/5/0.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.9620 +/- 0.2081 | 0.9453 +/- 0.0922 |  |  |
| `mlp_h64` | 1.0311 +/- 0.2152 | 1.0141 +/- 0.0964 |  |  |
| `mlp_h128` | 1.0409 +/- 0.2160 | 1.0186 +/- 0.0965 |  |  |
| `mlp_h64_64` | 1.0020 +/- 0.2117 | 0.9976 +/- 0.0957 |  |  |
| `upgd_low_noise` | 0.9621 +/- 0.2081 | 0.9434 +/- 0.0921 |  |  |
| `dynamic_sparse` | 0.9652 +/- 0.2075 | 0.9507 +/- 0.0914 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0400 +/- 0.0073; wins/losses/ties 10/0/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

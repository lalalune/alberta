# Step 2 Universal Portfolio

Protocol: 3 seeds, 1200 steps, final window 300, Hedge eta=8.0, discount=0.995, deployment policy=convex, retention router=class_imbalance, digits deployment objective=accuracy.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge; the router policies remain available as negative-control variants.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0386 +/- 0.0020 | 0.0529 +/- 0.0014 | 0.8700 +/- 0.0150 | 0.9029 +/- 0.0041 |
| `mlp_h64` | 0.0387 +/- 0.0020 | 0.0556 +/- 0.0013 | 0.8689 +/- 0.0144 | 0.9029 +/- 0.0041 |
| `mlp_h128` | 0.0425 +/- 0.0008 | 0.0577 +/- 0.0009 | 0.8522 +/- 0.0029 | 0.9054 +/- 0.0156 |
| `mlp_h64_64` | 0.0412 +/- 0.0017 | 0.0612 +/- 0.0009 | 0.8289 +/- 0.0111 | 0.8924 +/- 0.0170 |
| `upgd_low_noise` | 0.0570 +/- 0.0004 | 0.0739 +/- 0.0007 | 0.6900 +/- 0.0069 | 0.8404 +/- 0.0239 |
| `dynamic_sparse` | 0.0638 +/- 0.0013 | 0.0899 +/- 0.0006 | 0.6600 +/- 0.0067 | 0.8176 +/- 0.0183 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0001 +/- 0.0001; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: -0.0118 +/- 0.0118; wins/losses/ties 0/1/2.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0450 +/- 0.0005 | 0.0554 +/- 0.0014 | 0.8011 +/- 0.0097 | 0.8250 +/- 0.0229 |
| `mlp_h64` | 0.0514 +/- 0.0012 | 0.0609 +/- 0.0012 | 0.7789 +/- 0.0232 | 0.7811 +/- 0.0039 |
| `mlp_h128` | 0.0493 +/- 0.0012 | 0.0598 +/- 0.0006 | 0.7878 +/- 0.0166 | 0.8194 +/- 0.0192 |
| `mlp_h64_64` | 0.0535 +/- 0.0008 | 0.0650 +/- 0.0017 | 0.7433 +/- 0.0077 | 0.7730 +/- 0.0080 |
| `upgd_low_noise` | 0.0490 +/- 0.0016 | 0.0640 +/- 0.0003 | 0.7878 +/- 0.0244 | 0.8275 +/- 0.0195 |
| `dynamic_sparse` | 0.0596 +/- 0.0014 | 0.0805 +/- 0.0011 | 0.7456 +/- 0.0146 | 0.7792 +/- 0.0093 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0040 +/- 0.0007; wins/losses/ties 3/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0056 +/- 0.0075; wins/losses/ties 1/1/1.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.2762 +/- 0.0929 | 0.1791 +/- 0.0292 |  |  |
| `mlp_h64` | 0.2759 +/- 0.0935 | 0.1832 +/- 0.0298 |  |  |
| `mlp_h128` | 0.2781 +/- 0.0926 | 0.1797 +/- 0.0286 |  |  |
| `mlp_h64_64` | 0.3250 +/- 0.1132 | 0.2192 +/- 0.0368 |  |  |
| `upgd_low_noise` | 0.4057 +/- 0.1466 | 0.2545 +/- 0.0513 |  |  |
| `dynamic_sparse` | 0.3989 +/- 0.1519 | 0.2557 +/- 0.0518 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: -0.0011 +/- 0.0012; wins/losses/ties 2/1/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

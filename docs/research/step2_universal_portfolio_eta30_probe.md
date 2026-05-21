# Step 2 Universal Portfolio

Protocol: 10 seeds, 1200 steps, final window 300, Hedge eta=30.0, discount=0.995, deployment policy=convex, retention router=class_imbalance.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge; the router policies remain available as negative-control variants.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0389 +/- 0.0009 | 0.0556 +/- 0.0004 | 0.8657 +/- 0.0081 | 0.8911 +/- 0.0039 |
| `mlp_h64` | 0.0396 +/- 0.0008 | 0.0568 +/- 0.0006 | 0.8623 +/- 0.0075 | 0.8946 +/- 0.0029 |
| `mlp_h128` | 0.0435 +/- 0.0009 | 0.0590 +/- 0.0006 | 0.8447 +/- 0.0080 | 0.8959 +/- 0.0058 |
| `mlp_h64_64` | 0.0407 +/- 0.0011 | 0.0605 +/- 0.0007 | 0.8420 +/- 0.0110 | 0.8870 +/- 0.0053 |
| `upgd_low_noise` | 0.0567 +/- 0.0006 | 0.0745 +/- 0.0007 | 0.7003 +/- 0.0071 | 0.8369 +/- 0.0078 |
| `dynamic_sparse` | 0.0638 +/- 0.0012 | 0.0891 +/- 0.0007 | 0.6693 +/- 0.0155 | 0.8087 +/- 0.0084 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0001 +/- 0.0002; wins/losses/ties 6/1/3.
`test_accuracy` mixture-vs-best-MLP diff: -0.0124 +/- 0.0043; wins/losses/ties 0/6/4.

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0462 +/- 0.0014 | 0.0580 +/- 0.0007 | 0.8050 +/- 0.0096 | 0.8341 +/- 0.0139 |
| `mlp_h64` | 0.0493 +/- 0.0016 | 0.0606 +/- 0.0008 | 0.7873 +/- 0.0140 | 0.8174 +/- 0.0130 |
| `mlp_h128` | 0.0483 +/- 0.0012 | 0.0596 +/- 0.0007 | 0.8053 +/- 0.0084 | 0.8453 +/- 0.0109 |
| `mlp_h64_64` | 0.0522 +/- 0.0015 | 0.0646 +/- 0.0011 | 0.7570 +/- 0.0132 | 0.7998 +/- 0.0132 |
| `upgd_low_noise` | 0.0471 +/- 0.0013 | 0.0631 +/- 0.0005 | 0.7987 +/- 0.0146 | 0.8275 +/- 0.0129 |
| `dynamic_sparse` | 0.0586 +/- 0.0014 | 0.0797 +/- 0.0009 | 0.7337 +/- 0.0124 | 0.7857 +/- 0.0113 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0017 +/- 0.0003; wins/losses/ties 10/0/0.
`test_accuracy` mixture-vs-best-MLP diff: -0.0117 +/- 0.0093; wins/losses/ties 4/6/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.2743 +/- 0.0449 | 0.2163 +/- 0.0195 |  |  |
| `mlp_h64` | 0.2780 +/- 0.0456 | 0.2200 +/- 0.0199 |  |  |
| `mlp_h128` | 0.2733 +/- 0.0445 | 0.2174 +/- 0.0195 |  |  |
| `mlp_h64_64` | 0.3280 +/- 0.0513 | 0.2660 +/- 0.0239 |  |  |
| `upgd_low_noise` | 0.5051 +/- 0.1199 | 0.3650 +/- 0.0512 |  |  |
| `dynamic_sparse` | 0.4722 +/- 0.1073 | 0.3510 +/- 0.0479 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: -0.0019 +/- 0.0007; wins/losses/ties 1/8/1.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 1.5656 +/- 0.3542 | 1.3751 +/- 0.1360 |  |  |
| `mlp_h64` | 1.6235 +/- 0.3676 | 1.4328 +/- 0.1476 |  |  |
| `mlp_h128` | 1.6309 +/- 0.3676 | 1.4433 +/- 0.1496 |  |  |
| `mlp_h64_64` | 1.5961 +/- 0.3552 | 1.4058 +/- 0.1405 |  |  |
| `upgd_low_noise` | 1.6327 +/- 0.3528 | 1.4118 +/- 0.1405 |  |  |
| `dynamic_sparse` | 1.6312 +/- 0.3568 | 1.4115 +/- 0.1423 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0118 +/- 0.0127; wins/losses/ties 6/4/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

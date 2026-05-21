# Step 2 Universal Portfolio

Protocol: 10 seeds, 1200 steps, final window 300, Hedge eta=1.0, discount=0.995, deployment policy=convex, retention router=class_imbalance, digits deployment objective=mse.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge with a digits class-imbalance online MSE guard; the router policies remain available as negative-control variants.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0029 +/- 0.0001 | 0.0045 +/- 0.0000 | 0.9913 +/- 0.0005 | 0.2245 +/- 0.0104 |
| `mlp_h64` | 0.0052 +/- 0.0001 | 0.0091 +/- 0.0001 | 0.9847 +/- 0.0015 | 0.1195 +/- 0.0079 |
| `mlp_h128` | 0.0073 +/- 0.0001 | 0.0115 +/- 0.0001 | 0.9837 +/- 0.0015 | 0.1195 +/- 0.0081 |
| `mlp_h64_64` | 0.0029 +/- 0.0001 | 0.0062 +/- 0.0001 | 0.9913 +/- 0.0005 | 0.1006 +/- 0.0004 |
| `upgd_low_noise` | 0.0208 +/- 0.0007 | 0.0273 +/- 0.0004 | 0.9287 +/- 0.0043 | 0.2245 +/- 0.0104 |
| `dynamic_sparse` | 0.0271 +/- 0.0006 | 0.0397 +/- 0.0005 | 0.9317 +/- 0.0021 | 0.2135 +/- 0.0126 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0000 +/- 0.0000; wins/losses/ties 0/0/10.
`test_accuracy` mixture-vs-best-MLP diff: +0.0970 +/- 0.0114; wins/losses/ties 10/0/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

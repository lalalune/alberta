# Step 2 Universal Portfolio

Protocol: 3 seeds, 1200 steps, final window 300, Hedge eta=8.0, discount=0.995, deployment policy=mlp_convex, retention router=class_imbalance, digits deployment objective=mse.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge; the router policies remain available as negative-control variants.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.2762 +/- 0.0929 | 0.1791 +/- 0.0290 |  |  |
| `mlp_h64` | 0.2759 +/- 0.0935 | 0.1832 +/- 0.0298 |  |  |
| `mlp_h128` | 0.2781 +/- 0.0926 | 0.1797 +/- 0.0286 |  |  |
| `mlp_h64_64` | 0.3250 +/- 0.1132 | 0.2192 +/- 0.0368 |  |  |
| `upgd_low_noise` | 0.4057 +/- 0.1466 | 0.2545 +/- 0.0513 |  |  |
| `dynamic_sparse` | 0.3989 +/- 0.1519 | 0.2557 +/- 0.0518 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: -0.0011 +/- 0.0012; wins/losses/ties 2/1/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

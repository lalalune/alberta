# Step 2 Universal Portfolio

Protocol: 30 seeds, 1200 steps, final window 300, Hedge eta=8.0, discount=0.995, deployment policy=convex, retention router=class_imbalance, digits deployment objective=mse.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge; the router policies remain available as negative-control variants.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.2645 +/- 0.0198 | 0.2346 +/- 0.0128 |  |  |
| `mlp_h64` | 0.2659 +/- 0.0202 | 0.2376 +/- 0.0128 |  |  |
| `mlp_h128` | 0.2688 +/- 0.0200 | 0.2378 +/- 0.0130 |  |  |
| `mlp_h64_64` | 0.3196 +/- 0.0235 | 0.2912 +/- 0.0158 |  |  |
| `upgd_low_noise` | 0.5105 +/- 0.0503 | 0.4076 +/- 0.0299 |  |  |
| `dynamic_sparse` | 0.4873 +/- 0.0474 | 0.3958 +/- 0.0286 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: -0.0020 +/- 0.0006; wins/losses/ties 10/20/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

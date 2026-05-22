# Step 2 Universal Portfolio

Protocol: 10 seeds, 1200 steps, final window 300, Hedge eta=8.0, discount=0.995, deployment policy=mlp_selector, retention router=class_imbalance, digits deployment objective=mse.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge; the router policies remain available as negative-control variants.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.2745 +/- 0.0450 | 0.2163 +/- 0.0195 |  |  |
| `mlp_h64` | 0.2780 +/- 0.0456 | 0.2200 +/- 0.0199 |  |  |
| `mlp_h128` | 0.2733 +/- 0.0445 | 0.2174 +/- 0.0195 |  |  |
| `mlp_h64_64` | 0.3280 +/- 0.0513 | 0.2660 +/- 0.0239 |  |  |
| `upgd_low_noise` | 0.5051 +/- 0.1199 | 0.3650 +/- 0.0512 |  |  |
| `dynamic_sparse` | 0.4722 +/- 0.1073 | 0.3510 +/- 0.0479 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: -0.0021 +/- 0.0007; wins/losses/ties 1/8/1.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

# Step 2 Universal Portfolio

Protocol: 30 seeds, 1200 steps, final window 300, Hedge eta=1.0, discount=0.995, deployment policy=convex, retention router=class_imbalance, digits deployment objective=mse.

The runner tracks four causal deployment policies: all-expert convex Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only selector. The promoted default is all-expert convex Hedge with a digits class-imbalance online MSE guard; the router policies remain available as negative-control variants.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0029 +/- 0.0000 | 0.0046 +/- 0.0000 | 0.9910 +/- 0.0004 | 0.2386 +/- 0.0095 |
| `mlp_h64` | 0.0051 +/- 0.0001 | 0.0091 +/- 0.0001 | 0.9859 +/- 0.0007 | 0.1139 +/- 0.0044 |
| `mlp_h128` | 0.0072 +/- 0.0001 | 0.0116 +/- 0.0001 | 0.9852 +/- 0.0008 | 0.1132 +/- 0.0046 |
| `mlp_h64_64` | 0.0029 +/- 0.0000 | 0.0063 +/- 0.0001 | 0.9910 +/- 0.0004 | 0.1002 +/- 0.0003 |
| `upgd_low_noise` | 0.0205 +/- 0.0004 | 0.0267 +/- 0.0003 | 0.9313 +/- 0.0022 | 0.2386 +/- 0.0095 |
| `dynamic_sparse` | 0.0271 +/- 0.0004 | 0.0399 +/- 0.0003 | 0.9296 +/- 0.0017 | 0.2213 +/- 0.0097 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0000 +/- 0.0000; wins/losses/ties 0/0/30.
`test_accuracy` mixture-vs-best-MLP diff: +0.1203 +/- 0.0085; wins/losses/ties 30/0/0.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 1.6413 +/- 0.1943 | 1.3549 +/- 0.0803 |  |  |
| `mlp_h64` | 1.7138 +/- 0.2073 | 1.4161 +/- 0.0849 |  |  |
| `mlp_h128` | 1.7318 +/- 0.2090 | 1.4270 +/- 0.0854 |  |  |
| `mlp_h64_64` | 1.6911 +/- 0.2030 | 1.4060 +/- 0.0827 |  |  |
| `upgd_low_noise` | 1.6932 +/- 0.1985 | 1.3957 +/- 0.0824 |  |  |
| `dynamic_sparse` | 1.6969 +/- 0.2011 | 1.3992 +/- 0.0837 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0187 +/- 0.0137; wins/losses/ties 15/15/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

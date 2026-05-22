# Step 2 Universal Portfolio

Protocol: 1 seeds, 80 steps, final window 20, Hedge eta=8.0, discount=0.995, retention router=class_imbalance.

The live portfolio keeps two causal routers: MLP-only Hedge as a fallback and all-expert Hedge over `mlp_h64`, `mlp_h128`, `mlp_h64_64`, `upgd_low_noise`, and `dynamic_sparse`. A causal EMA guard deploys the all-expert prediction only when its prior loss trace does not trail the MLP-only fallback.

## digits_iid

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0878 +/- 0.0000 | 0.0925 +/- 0.0000 | 0.6000 +/- 0.0000 | 0.7124 +/- 0.0000 |
| `mlp_h64` | 0.0966 +/- 0.0000 | 0.1064 +/- 0.0000 | 0.4000 +/- 0.0000 | 0.6438 +/- 0.0000 |
| `mlp_h128` | 0.0935 +/- 0.0000 | 0.1001 +/- 0.0000 | 0.5500 +/- 0.0000 | 0.6994 +/- 0.0000 |
| `mlp_h64_64` | 0.1132 +/- 0.0000 | 0.1118 +/- 0.0000 | 0.2000 +/- 0.0000 | 0.4750 +/- 0.0000 |
| `upgd_low_noise` | 0.1142 +/- 0.0000 | 0.1446 +/- 0.0000 | 0.2500 +/- 0.0000 | 0.3284 +/- 0.0000 |
| `dynamic_sparse` | 0.1427 +/- 0.0000 | 0.1824 +/- 0.0000 | 0.1000 +/- 0.0000 | 0.2430 +/- 0.0000 |

`final_window_mse` mixture-vs-best-MLP diff: +0.0057 +/- 0.0000; wins/losses/ties 1/0/0.
`test_accuracy` mixture-vs-best-MLP diff: +0.0130 +/- 0.0000; wins/losses/ties 1/0/0.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.7870 +/- 0.0000 | 0.5926 +/- 0.0000 |  |  |
| `mlp_h64` | 0.8146 +/- 0.0000 | 0.6061 +/- 0.0000 |  |  |
| `mlp_h128` | 0.8116 +/- 0.0000 | 0.6141 +/- 0.0000 |  |  |
| `mlp_h64_64` | 0.9424 +/- 0.0000 | 0.6578 +/- 0.0000 |  |  |
| `upgd_low_noise` | 0.7870 +/- 0.0000 | 0.5721 +/- 0.0000 |  |  |
| `dynamic_sparse` | 0.8204 +/- 0.0000 | 0.5995 +/- 0.0000 |  |  |

`final_window_mse` mixture-vs-best-MLP diff: +0.0246 +/- 0.0000; wins/losses/ties 1/0/0.

## Assessment Rule

Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, the difference is best MLP minus mixture; for accuracy, it is mixture minus best MLP. A universal Step 2 claim still requires this result to hold on the full matrix with enough seeds and no retained-test regression.

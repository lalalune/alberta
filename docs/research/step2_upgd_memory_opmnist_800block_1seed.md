# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `48000000`
- Seeds: `1`
- Permutations: `800`
- Task block size: `60000`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.015042 +/- 0.000000 | 0.904000 +/- 0.000000 | 0.136308 +/- 0.000000 | 0.123512 +/- 0.000000 |
| `mlp_h64` | 0.027254 +/- 0.000000 | 0.885200 +/- 0.000000 | 0.104611 +/- 0.000000 | 0.128668 +/- 0.000000 |
| `mlp_h128` | 0.021807 +/- 0.000000 | 0.901800 +/- 0.000000 | 0.100162 +/- 0.000000 | 0.129090 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h128`: +0.010877 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h128`: +0.013930 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h128`: +0.006765 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.002200 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h128`: -0.036147 +/- 0.000000; wins/losses/ties 0/1/0.
- `test_accuracy` vs `mlp_h128`: -0.005578 +/- 0.000000; wins/losses/ties 0/1/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

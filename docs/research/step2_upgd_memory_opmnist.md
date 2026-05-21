# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_28x28`
- Steps: `1000`
- Seeds: `1`
- Permutations: `5`
- Task block size: `200`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.026111 +/- 0.000000 | 0.825000 +/- 0.000000 | 0.011998 +/- 0.000000 | 0.933500 +/- 0.000000 |
| `mlp_h64` | 0.074930 +/- 0.000000 | 0.495000 +/- 0.000000 | 0.072015 +/- 0.000000 | 0.538000 +/- 0.000000 |
| `mlp_h128` | 0.087778 +/- 0.000000 | 0.500000 +/- 0.000000 | 0.087226 +/- 0.000000 | 0.398000 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.049370 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.325000 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h64`: +0.048819 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.325000 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h64`: +0.060017 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h64`: +0.395500 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

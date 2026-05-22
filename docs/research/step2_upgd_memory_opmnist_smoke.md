# Step 2 UPGD-Memory OPMNIST

This note records the resumable single-learner OPMNIST run for the packaged UPGD-memory trace learner against fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_28x28`
- Steps: `20`
- Seeds: `1`
- Permutations: `2`
- Task block size: `10`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.089589 +/- 0.000000 | 0.400000 +/- 0.000000 | 0.073571 +/- 0.000000 | 0.516667 +/- 0.000000 |
| `mlp_h64` | 0.183152 +/- 0.000000 | 0.000000 +/- 0.000000 | 0.184402 +/- 0.000000 | 0.116667 +/- 0.000000 |
| `mlp_h128` | 0.373220 +/- 0.000000 | 0.000000 +/- 0.000000 | 0.357574 +/- 0.000000 | 0.100000 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.072187 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.200000 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h64`: +0.093563 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.400000 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h64`: +0.110831 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h64`: +0.400000 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

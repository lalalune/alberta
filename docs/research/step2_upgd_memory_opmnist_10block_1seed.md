# Step 2 UPGD-Memory OPMNIST

This note records the resumable single-learner OPMNIST run for the packaged UPGD-memory trace learner against fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `600000`
- Seeds: `1`
- Permutations: `800`
- Task block size: `60000`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.008422 +/- 0.000000 | 0.948400 +/- 0.000000 | 0.051540 +/- 0.000000 | 0.650970 +/- 0.000000 |
| `mlp_h64` | 0.017371 +/- 0.000000 | 0.931200 +/- 0.000000 | 0.077128 +/- 0.000000 | 0.398200 +/- 0.000000 |
| `mlp_h128` | 0.016182 +/- 0.000000 | 0.936800 +/- 0.000000 | 0.075334 +/- 0.000000 | 0.417370 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h128`: +0.010216 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h128`: +0.013312 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h128`: +0.007760 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.011600 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h128`: +0.023794 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h128`: +0.233600 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

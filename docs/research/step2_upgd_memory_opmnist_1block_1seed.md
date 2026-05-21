# Step 2 UPGD-Memory OPMNIST

This note records the resumable single-learner OPMNIST run for the packaged UPGD-memory trace learner against fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `60000`
- Seeds: `1`
- Permutations: `800`
- Task block size: `60000`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.010051 +/- 0.000000 | 0.934400 +/- 0.000000 | 0.009361 +/- 0.000000 | 0.938700 +/- 0.000000 |
| `mlp_h64` | 0.018119 +/- 0.000000 | 0.928200 +/- 0.000000 | 0.017901 +/- 0.000000 | 0.931000 +/- 0.000000 |
| `mlp_h128` | 0.017624 +/- 0.000000 | 0.928400 +/- 0.000000 | 0.016415 +/- 0.000000 | 0.936500 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.009995 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.011733 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h128`: +0.007573 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.006000 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h128`: +0.007054 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h128`: +0.002200 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

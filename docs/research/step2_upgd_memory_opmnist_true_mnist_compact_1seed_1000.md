# Step 2 UPGD-Memory OPMNIST

This note records the resumable single-learner OPMNIST run for the packaged UPGD-memory trace learner against fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `1000`
- Seeds: `1`
- Permutations: `5`
- Task block size: `200`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.050627 +/- 0.000000 | 0.660000 +/- 0.000000 | 0.045636 +/- 0.000000 | 0.694400 +/- 0.000000 |
| `mlp_h64` | 0.078163 +/- 0.000000 | 0.495000 +/- 0.000000 | 0.082736 +/- 0.000000 | 0.354800 +/- 0.000000 |
| `mlp_h128` | 0.093170 +/- 0.000000 | 0.450000 +/- 0.000000 | 0.090014 +/- 0.000000 | 0.352000 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.034929 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.189000 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h64`: +0.027536 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.165000 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h64`: +0.037100 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h64`: +0.339600 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

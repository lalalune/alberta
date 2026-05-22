# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `480000`
- Seeds: `1`
- Permutations: `800`
- Task block size: `60000`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.008751 +/- 0.000000 | 0.944800 +/- 0.000000 | 0.044436 +/- 0.000000 | 0.693912 +/- 0.000000 |
| `mlp_h64` | 0.017911 +/- 0.000000 | 0.931200 +/- 0.000000 | 0.083785 +/- 0.000000 | 0.425162 +/- 0.000000 |
| `mlp_h128` | 0.016612 +/- 0.000000 | 0.937200 +/- 0.000000 | 0.086591 +/- 0.000000 | 0.403412 +/- 0.000000 |
| `upgd_structure_linear_h64` | 0.016883 +/- 0.000000 | 0.928400 +/- 0.000000 | 0.060148 +/- 0.000000 | 0.575850 +/- 0.000000 |
| `upgd_structure_softmax_h64` | 0.008751 +/- 0.000000 | 0.945600 +/- 0.000000 | 0.051763 +/- 0.000000 | 0.674312 +/- 0.000000 |
| `upgd_structure_linear_h128` | 0.015402 +/- 0.000000 | 0.935800 +/- 0.000000 | 0.054997 +/- 0.000000 | 0.626437 +/- 0.000000 |
| `upgd_structure_softmax_h128` | 0.008232 +/- 0.000000 | 0.949000 +/- 0.000000 | 0.048247 +/- 0.000000 | 0.697887 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h128`: +0.010148 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h128`: +0.012615 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h128`: +0.007861 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.007600 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h64`: +0.039349 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h64`: +0.268750 +/- 0.000000; wins/losses/ties 1/0/0.

## Additional Candidate vs Best MLP

- `upgd_structure_linear_h64` `online_mean_mse` vs `mlp_h128`: +0.000969 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64` `online_mean_mse` vs `mlp_h128`: +0.009780 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h128` `online_mean_mse` vs `mlp_h128`: +0.002503 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `online_mean_mse` vs `mlp_h128`: +0.010215 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h64` `online_mean_accuracy` vs `mlp_h128`: -0.006504 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h64` `online_mean_accuracy` vs `mlp_h128`: +0.011373 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h128` `online_mean_accuracy` vs `mlp_h128`: +0.000221 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `online_mean_accuracy` vs `mlp_h128`: +0.014927 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h64` `final_window_mse` vs `mlp_h128`: -0.000270 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h64` `final_window_mse` vs `mlp_h128`: +0.007862 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h128` `final_window_mse` vs `mlp_h128`: +0.001211 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `final_window_mse` vs `mlp_h128`: +0.008380 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h64` `final_window_accuracy` vs `mlp_h128`: -0.008800 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h64` `final_window_accuracy` vs `mlp_h128`: +0.008400 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h128` `final_window_accuracy` vs `mlp_h128`: -0.001400 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128` `final_window_accuracy` vs `mlp_h128`: +0.011800 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h64` `test_mse` vs `mlp_h64`: +0.023637 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64` `test_mse` vs `mlp_h64`: +0.032022 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h128` `test_mse` vs `mlp_h64`: +0.028787 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `test_mse` vs `mlp_h64`: +0.035538 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h64` `test_accuracy` vs `mlp_h64`: +0.150687 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64` `test_accuracy` vs `mlp_h64`: +0.249150 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h128` `test_accuracy` vs `mlp_h64`: +0.201275 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `test_accuracy` vs `mlp_h64`: +0.272725 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

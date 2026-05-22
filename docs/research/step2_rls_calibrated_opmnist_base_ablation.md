# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_8x8`
- Steps: `600`
- Seeds: `3`
- Permutations: `5`
- Task block size: `120`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `mlp_h64` | 0.081529 +/- 0.001249 | 0.420000 +/- 0.022546 | 0.088188 +/- 0.002619 | 0.326333 +/- 0.029180 |
| `upgd_structure_softmax_h64` | 0.077259 +/- 0.002255 | 0.365000 +/- 0.033292 | 0.070406 +/- 0.001369 | 0.455000 +/- 0.010970 |
| `upgd_structure_softmax_h64_rls_cal` | 0.078665 +/- 0.000918 | 0.358333 +/- 0.016915 | 0.071981 +/- 0.001877 | 0.429000 +/- 0.047318 |

## Primary vs Best MLP


## Additional Candidate vs Best MLP

- `upgd_structure_softmax_h64` `online_mean_mse` vs `mlp_h64`: +0.013515 +/- 0.001902; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_rls_cal` `online_mean_mse` vs `mlp_h64`: +0.013066 +/- 0.001109; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `online_mean_accuracy` vs `mlp_h64`: -0.010556 +/- 0.010556; wins/losses/ties 1/2/0.
- `upgd_structure_softmax_h64_rls_cal` `online_mean_accuracy` vs `mlp_h64`: -0.008333 +/- 0.015275; wins/losses/ties 2/1/0.
- `upgd_structure_softmax_h64` `final_window_mse` vs `mlp_h64`: +0.004270 +/- 0.001766; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_rls_cal` `final_window_mse` vs `mlp_h64`: +0.002864 +/- 0.000538; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `final_window_accuracy` vs `mlp_h64`: -0.055000 +/- 0.032787; wins/losses/ties 1/2/0.
- `upgd_structure_softmax_h64_rls_cal` `final_window_accuracy` vs `mlp_h64`: -0.061667 +/- 0.016415; wins/losses/ties 0/3/0.
- `upgd_structure_softmax_h64` `test_mse` vs `mlp_h64`: +0.017782 +/- 0.002686; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_rls_cal` `test_mse` vs `mlp_h64`: +0.016207 +/- 0.000755; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `test_accuracy` vs `mlp_h64`: +0.128667 +/- 0.018224; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_rls_cal` `test_accuracy` vs `mlp_h64`: +0.102667 +/- 0.035366; wins/losses/ties 3/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

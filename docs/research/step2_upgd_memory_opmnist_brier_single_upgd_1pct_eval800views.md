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
| `mlp_h64` | 0.018059 +/- 0.000000 | 0.927600 +/- 0.000000 | 0.098090 +/- 0.000000 | 0.108986 +/- 0.000000 |
| `mlp_h128` | 0.016609 +/- 0.000000 | 0.938800 +/- 0.000000 | 0.108615 +/- 0.000000 | 0.105385 +/- 0.000000 |
| `upgd_structure_brier_h64` | 0.008780 +/- 0.000000 | 0.942000 +/- 0.000000 | 0.140596 +/- 0.000000 | 0.111268 +/- 0.000000 |
| `upgd_structure_brier_h128` | 0.007589 +/- 0.000000 | 0.950400 +/- 0.000000 | 0.141606 +/- 0.000000 | 0.109963 +/- 0.000000 |

## Primary vs Best MLP


## Additional Candidate vs Best MLP

- `upgd_structure_brier_h64` `online_mean_mse` vs `mlp_h128`: +0.009891 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_brier_h128` `online_mean_mse` vs `mlp_h128`: +0.010506 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_brier_h64` `online_mean_accuracy` vs `mlp_h128`: +0.008531 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_brier_h128` `online_mean_accuracy` vs `mlp_h128`: +0.012731 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_brier_h64` `final_window_mse` vs `mlp_h128`: +0.007829 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_brier_h128` `final_window_mse` vs `mlp_h128`: +0.009019 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_brier_h64` `final_window_accuracy` vs `mlp_h128`: +0.003200 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_brier_h128` `final_window_accuracy` vs `mlp_h128`: +0.011600 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_brier_h64` `test_mse` vs `mlp_h64`: -0.042506 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_brier_h128` `test_mse` vs `mlp_h64`: -0.043516 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_brier_h64` `test_accuracy` vs `mlp_h64`: +0.002282 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_brier_h128` `test_accuracy` vs `mlp_h64`: +0.000977 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
